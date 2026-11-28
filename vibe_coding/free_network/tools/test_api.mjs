// Runs the real worker on a throwaway local database and checks the promises the network makes.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { createDecipheriv, createECDH, createPublicKey, generateKeyPairSync, hkdfSync, randomBytes, verify } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { deviceName } from '../src/log.js';
import { join } from 'node:path';

// Other projects' servers sometimes sit on these ports. TEST_PORT_BASE=18792 npm test moves them all.
const BASE = Number(process.env.TEST_PORT_BASE ?? 8792);
const PORT = BASE + 7;
const PROD_PORT = BASE + 6;
const STORAGE_PORT = BASE + 3;
const UPLOADS_PORT = BASE + 2;
const INVITE_PORT = BASE + 1;
const YOUTUBE_PORT = BASE;
const VIDEO_ID = 'abcdefghijk';
const youtubeRequests = [];
const digestRequests = [];
const pictureRequests = [];
const mailSent = [];
const revoked = [];
const peopleRequests = [];
const tileRequests = [];
const pushes = [];

// The key the test network signs its pushes with, as the two strings the worker is given.
const vapid = (() => {
  const jwk = generateKeyPairSync('ec', { namedCurve: 'P-256' }).privateKey.export({ format: 'jwk' });
  const point = Buffer.concat([Buffer.from([4]), Buffer.from(jwk.x, 'base64url'), Buffer.from(jwk.y, 'base64url')]);
  return { public: point.toString('base64url'), private: jwk.d, jwk: { kty: 'EC', crv: 'P-256', x: jwk.x, y: jwk.y } };
})();

// Opens a sealed push the way a browser does (RFC 8291), with the private key of the device.
function openPush(ecdh, auth, body) {
  const salt = body.subarray(0, 16);
  const sender = body.subarray(21, 21 + body[20]);
  const sealed = body.subarray(21 + body[20]);
  const secret = hkdfSync('sha256', ecdh.computeSecret(sender), auth, Buffer.concat([Buffer.from('WebPush: info\0'), ecdh.getPublicKey(), sender]), 32);
  const opener = createDecipheriv('aes-128-gcm', Buffer.from(hkdfSync('sha256', secret, salt, 'Content-Encoding: aes128gcm\0', 16)),
    Buffer.from(hkdfSync('sha256', secret, salt, 'Content-Encoding: nonce\0', 12)));
  opener.setAuthTag(sealed.subarray(-16));
  const plain = Buffer.concat([opener.update(sealed.subarray(0, -16)), opener.final()]);
  return plain.subarray(0, plain.lastIndexOf(2)).toString();
}

// A device that subscribed to pushes: what its browser hands to the network, and a way to wait
// for the pushes that the fake push service received for it and read them.
function device(name) {
  const ecdh = createECDH('prime256v1');
  ecdh.generateKeys();
  const auth = randomBytes(16);
  const subscription = { endpoint: `http://localhost:${YOUTUBE_PORT}/push/${name}`, keys: { p256dh: ecdh.getPublicKey().toString('base64url'), auth: auth.toString('base64url') } };
  const got = () => pushes.filter((push) => push.name === name);
  return {
    subscription,
    count: () => got().length,
    async wait(n) {
      for (let i = 0; i < 100 && got().length < n; i++) await new Promise((resolve) => setTimeout(resolve, 50));
      assert.equal(got().length, n, `pushes to ${name}`);
      return got().map((push) => ({ ...push, message: JSON.parse(openPush(ecdh, auth, push.body)) }));
    },
  };
}

// Stands in for YouTube, and records what the worker sends it.
const fakeYoutube = createServer((request, response) => {
  if (request.url === '/token') {
    let body = '';
    request.on('data', (part) => { body += part; });
    request.on('end', () => {
      // A code like "google-x" is a person with no address; "google-x:a@b.com" has one, "!" unverified.
      // A third part is the picture of the account: "photo" is a real one, "letter" is the letter
      // Google draws when there is none, and "elsewhere" points outside Google.
      const form = new URLSearchParams(body);
      // The key that lets the network send mail is traded for a pass, unless Google withdrew it.
      if (form.get('grant_type') === 'refresh_token') {
        const gone = form.get('refresh_token').includes('withdrawn');
        response.statusCode = gone ? 400 : 200;
        response.setHeader('Content-Type', 'application/json');
        response.end(JSON.stringify(gone ? { error: 'invalid_grant' } : { access_token: `pass-for-${form.get('refresh_token')}` }));
        return;
      }
      const [code, email, picture] = form.get('code').split(':');
      const claims = { iss: 'https://accounts.google.com', aud: code === 'wrong-audience' ? 'other' : 'test-client', sub: code };
      if (email) Object.assign(claims, { email: email.replace('!', ''), email_verified: !email.includes('!'), name: 'שם מגוגל' });
      if (picture) claims.picture = picture === 'elsewhere' ? 'http://127.0.0.1:1/a-/photo=s96-c' : `http://localhost:${YOUTUBE_PORT}/a-/${picture}=s96-c`;
      const encoded = Buffer.from(JSON.stringify(claims)).toString('base64url');
      response.setHeader('Content-Type', 'application/json');
      // Google hands a lasting key only when asked for one, and the scopes the person agreed to.
      // "contacts" in the code is a person who also ticked the reading of their contacts.
      const scope = `openid email${code.includes('unticked') ? '' : ' https://www.googleapis.com/auth/gmail.send'}${code.includes('contacts')
        ? ' https://www.googleapis.com/auth/contacts.readonly https://www.googleapis.com/auth/contacts.other.readonly' : ''}`;
      response.end(JSON.stringify({ id_token: `x.${encoded}.y`, refresh_token: `key-of-${code}`, scope }));
    });
    return;
  }
  // The sources of the update: two Wikipedias, the Pardespedia events board and the building plans.
  // The Israeli channels: two with pictures, one through a headline service, one that fails.
  if (request.url.startsWith('/feed/')) {
    digestRequests.push(request.url);
    const channel = request.url.slice(6, -4);
    const item = (title, minutes, extra = '') => `<item><title>${title}</title><link>https://${channel}.example/${minutes}</link>
<pubDate>${new Date(Date.now() - minutes * 60000).toUTCString()}</pubDate>${extra}</item>`;
    const feeds = {
      n12: item('ידיעה ישנה ב-N12', 90) + item('ידיעה חדשה ב-N12 &amp; עוד', 5, '<image>https://img.mako.co.il/a.jpg</image>'),
      c14: item('<![CDATA[כותרת בערוץ 14]]>', 10, '<content:encoded><![CDATA[<p><img src="https://www.c14.co.il/b.jpg"></p>]]></content:encoded>'),
      kan: item('כותרת בכאן - כאן 11', 20, '<source url="https://www.kan.org.il">כאן 11</source>') + item('כתובת לא בטוחה', 1).replace('https://kan.example', 'javascript:alert(1)//'),
    };
    if (!feeds[channel]) {
      response.statusCode = 403;
      response.end();
      return;
    }
    response.setHeader('Content-Type', 'application/rss+xml');
    response.end(`<?xml version="1.0"?><rss><channel>${feeds[channel]}</channel></rss>`);
    return;
  }
  if (request.url.startsWith('/w/api.php') || request.url.startsWith('/api.php') || request.url === '/plans.json') {
    digestRequests.push(request.url);
    const today = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Jerusalem' }).format(Date.now());
    const [, month, day] = today.split('-').map(Number);
    const heMonth = ['ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר'][month - 1];
    const card = (title) => `<div style="font-size:125%; font-weight:bold; line-height:1.3;">${title}</div>
<div style="margin-top:3px;"><span style="display:inline-block; width:1.7em;">🕗</span>20:30</div>
<div style="margin-top:3px;"><span style="display:inline-block; width:1.7em;">📍</span>[https://maps.example/?q=1 בית הראשונים]</div>`;
    const body = request.url === '/plans.json' ? { plans: [
      { num: '308-1', name: 'תוכנית', address: 'דרך הבנים', adds: '12 יחידות דיור', shuts: '2999-01-01', hidden: false },
      { num: '308-2', name: 'נסגרה', shuts: '2000-01-01', hidden: false },
      { num: '308-3', name: 'מוסתרת', shuts: '2999-01-01', hidden: true }] }
      : request.url.startsWith('/api.php') ? { parse: { wikitext: `== אירועים ==
<!-- AUTO:START -->
<div style="font-size:120%; border-bottom:2px solid #e3bcac; margin:22px 0 4px;">יום כלשהו, ${day} ב${heMonth} · היום</div>
<div>[[קובץ:ערב ג'אז.jpg|200px|ערב ג'אז]]</div>
${card('[https://events.example/1 ערב ג\'אז]')}
${card('[https://events.example/2 הופעה] <span>בוטל</span>')}
<div style="font-size:120%; border-bottom:2px solid #e3bcac; margin:22px 0 4px;">יום רחוק, ${day} ב${heMonth}</div>
${card('מאוחר יותר באותו יום')}
<!-- AUTO:END -->
<div style="font-size:125%; font-weight:bold; line-height:1.3;">מחוץ ללוח</div>` } }
        : request.url.includes('prop=pageimages')
          ? { query: { normalized: [{ from: 'גשם', to: 'גשם' }], pages: [
            { title: 'גשם', thumbnail: { source: 'https://upload.wikimedia.org/rain.jpg' } },
            { title: 'Big Wave', thumbnail: { source: 'https://evil.example/wave.jpg' } }] } }
        : request.url.includes('In_the_news')
          ? { parse: { text: '<div><ul><li>The <a href="/wiki/Sea">sea</a> sees <b><a href="/wiki/Big_Wave" title="x">a big wave</a></b> <i>(pictured)</i>.</li></ul><ul><li>Ongoing</li></ul></div>' } }
          : { parse: { text: '<div><ul><li><a href="/wiki/%D7%92%D7%A9%D7%9D" title="גשם">גשם</a> <small>(בתמונה)</small> ירד &amp; ירד ב<a href="/wiki/X">צפון</a> (<a rel="nofollow" class="external text" href="https://news.example/1">אתר</a>)</li>\n<li>ידיעה שנייה</li></ul></div>' } };
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify(body));
    return;
  }
  // Google's contacts: saved ones in two pages, and the people remembered from past letters.
  // A key with "disabled" in it meets a project where the People API was never switched on.
  if (request.url.startsWith('/v1/people/me/connections') || request.url.startsWith('/v1/otherContacts')) {
    const query = new URL(request.url, 'http://x').searchParams;
    peopleRequests.push({ url: request.url, pass: request.headers.authorization });
    response.setHeader('Content-Type', 'application/json');
    const person = (name, ...emails) => ({ ...(name ? { names: [{ displayName: name }] } : {}), emailAddresses: emails.map((value) => ({ value })) });
    if (request.headers.authorization.includes('disabled')) {
      response.statusCode = 403;
      response.end(JSON.stringify({ error: { code: 403, status: 'PERMISSION_DENIED', details: [{ reason: 'SERVICE_DISABLED' }] } }));
    } else if (request.url.startsWith('/v1/otherContacts')) {
      const fine = query.get('readMask') === 'names,emailAddresses';
      response.end(JSON.stringify(fine ? { otherContacts: [person(null, 'Dana@Example.org'), person('רון שכתבתי לו', 'ron@example.org'), person('בלי כתובת'), person('שבור', 'not-an-address')] } : {}));
    } else if (query.get('personFields') !== 'names,emailAddresses') {
      response.end('{}');
    } else if (!query.get('pageToken')) {
      response.end(JSON.stringify({ connections: [person('דנה כהן', 'dana@example.org', 'dana.work@example.com')], nextPageToken: 'page 2' }));
    } else {
      response.end(JSON.stringify({ connections: query.get('pageToken') === 'page 2' ? [person('יוסי לוי', 'yossi@example.org')] : [] }));
    }
    return;
  }
  if (request.url === '/gmail/v1/users/me/messages/send' || request.url === '/revoke') {
    let body = '';
    request.on('data', (part) => { body += part; });
    request.on('end', () => {
      if (request.url === '/revoke') revoked.push(new URLSearchParams(body).get('token'));
      else mailSent.push({ pass: request.headers.authorization, message: Buffer.from(JSON.parse(body).raw, 'base64url').toString() });
      response.setHeader('Content-Type', 'application/json');
      response.end('{}');
    });
    return;
  }
  if (request.url.startsWith('/a-/')) {
    pictureRequests.push(request.url);
    if (request.url.startsWith('/a-/photo')) response.end(phoneJpeg());
    else response.end(Buffer.from('\x89PNG letter'));
    return;
  }
  // The aerial pictures of the globe: one tile exists, one address answers with a page of text.
  if (request.url.startsWith('/ArcGIS/')) {
    tileRequests.push(request);
    if (request.url.endsWith('/tile/12/1664/2445')) {
      response.setHeader('Content-Type', 'image/jpeg');
      response.end(Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
    } else if (request.url.endsWith('/tile/12/1664/2446')) {
      response.setHeader('Content-Type', 'text/html');
      response.end('<h1>service is down</h1>');
    } else {
      response.statusCode = 404;
      response.end();
    }
    return;
  }
  // A push service: it keeps what it was sent, and one address says the device has unsubscribed.
  if (request.url.startsWith('/push/')) {
    const parts = [];
    request.on('data', (part) => parts.push(part));
    request.on('end', () => {
      const name = request.url.slice(6);
      pushes.push({ name, headers: request.headers, body: Buffer.concat(parts) });
      response.statusCode = name === 'gone' ? 410 : 201;
      response.end();
    });
    return;
  }
  // Web pages to set in a post: one that tells its title and picture, and two that refuse frames.
  if (request.url.startsWith('/site/')) {
    const name = request.url.slice(6);
    if (name === 'cover.png') {
      response.setHeader('Content-Type', 'image/png');
      response.end(Buffer.from([0x89, 0x50, 0x4e, 0x47]));
      return;
    }
    if (name === 'denied') response.setHeader('X-Frame-Options', 'DENY');
    if (name === 'own-only') response.setHeader('Content-Security-Policy', "default-src 'self'; frame-ancestors 'self'");
    if (name === 'anyone') {
      response.setHeader('X-Frame-Options', 'DENY');
      response.setHeader('Content-Security-Policy', 'frame-ancestors *');
    }
    response.setHeader('Content-Type', 'text/html; charset=utf-8');
    response.end(`<!doctype html><html><head><title>כותרת &amp; משנה</title>
      <meta content="מעבדה שאפשר לשחק בה" property="og:description">
      <meta property="og:image" content="/site/cover.png"></head><body><script>1</script></body></html>`);
    return;
  }
  youtubeRequests.push(request);
  if (request.url.startsWith('/oembed') && request.url.includes(VIDEO_ID)) {
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify({ title: 'סרטון לדוגמה' }));
  } else if (request.url === `/vi/${VIDEO_ID}/hqdefault.jpg`) {
    response.end(Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
  } else {
    response.statusCode = 404;
    response.end();
  }
});
const root = mkdtempSync(join(tmpdir(), 'free-network-test-'));
const servers = [];

// Each server needs its own database directory: two cannot share one.
function startServer(port, extraArgs) {
  const state = join(root, String(port));
  const quiet = { stdio: 'ignore', env: { ...process.env, CI: '1' } };
  execFileSync('npx', ['wrangler', 'd1', 'migrations', 'apply', 'free-network', '--local', '--persist-to', state], quiet);
  const server = spawn('npx', ['wrangler', 'dev', '--port', String(port), '--inspector-port', String(port + 1000),
    '--persist-to', state, ...extraArgs], { stdio: 'ignore', detached: true });
  servers.push(server);
}

async function waitFor(port) {
  for (let i = 0; i < 120; i++) {
    try {
      await fetch(`http://localhost:${port}/api/me`);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw new Error(`server on ${port} did not start`);
}

function client(port = PORT) {
  let cookie = '';
  const call = async function call(method, path, body) {
    const response = await fetch(`http://localhost:${port}/api${path}`, {
      method,
      headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(cookie ? { Cookie: cookie } : {}) },
      body: Buffer.isBuffer(body) ? body : body ? JSON.stringify(body) : undefined,
    });
    const setCookie = response.headers.get('set-cookie');
    if (setCookie) cookie = setCookie.split(';')[0];
    const bytes = await response.arrayBuffer();
    const isJson = response.headers.get('content-type')?.includes('json');
    return { status: response.status, bytes, data: isJson ? JSON.parse(Buffer.from(bytes)) : null };
  };
  call.cookie = () => cookie;
  return call;
}

async function signUp(name, port = PORT) {
  const call = client(port);
  const { data } = await call('POST', '/auth/request', { email: `${name}@example.com` });
  const token = data.devLink.split('#token=')[1];
  assert.equal((await call('POST', '/auth/verify', { token })).status, 200);
  await call('PUT', '/me', { name });
  const { data: { user } } = await call('GET', '/me');
  return { call, id: user.id, token };
}

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

let alice, bob, carol, anon;

test('login token works once only', async () => {
  alice = await signUp('alice');
  bob = await signUp('bob');
  carol = await signUp('carol');
  anon = client();
  assert.equal((await anon('POST', '/auth/verify', { token: alice.token })).status, 400);
});

test('login requests are rate limited per email', async () => {
  const call = client();
  for (let i = 0; i < 5; i++) assert.equal((await call('POST', '/auth/request', { email: 'x@example.com' })).status, 200);
  assert.equal((await call('POST', '/auth/request', { email: 'X@example.com' })).status, 429);
});

test('friend request becomes friendship and the audience count follows', async () => {
  assert.equal((await alice.call('GET', '/audience')).data.friends, 0);
  assert.equal((await alice.call('POST', `/friends/${bob.id}`)).data.status, 'pending');
  assert.equal((await alice.call('GET', '/audience')).data.friends, 0);
  assert.equal((await bob.call('POST', `/friends/${alice.id}`)).data.status, 'accepted');
  assert.equal((await alice.call('GET', '/audience')).data.friends, 1);
  assert.equal((await bob.call('GET', '/audience')).data.friends, 1);
});

let friendsPost, publicPost;

test('a friend gets every post, newest first, with no filtering', async () => {
  const ids = [];
  for (let i = 0; i < 5; i++) {
    ids.push((await alice.call('POST', '/posts', { body: `פוסט ${i}`, audience: 'friends' })).data.id);
  }
  friendsPost = ids[0];
  publicPost = (await alice.call('POST', '/posts', { body: 'לכולם', audience: 'public' })).data.id;
  const feed = (await bob.call('GET', '/feed')).data.posts.map((post) => post.id);
  assert.deepEqual(feed, [publicPost, ...ids.reverse()]);
});

test('friends posts stay private from strangers and logged out visitors', async () => {
  assert.equal((await carol.call('GET', '/feed')).data.posts.length, 0);
  assert.equal((await carol.call('GET', `/posts/${friendsPost}`)).status, 404);
  assert.equal((await anon('GET', `/posts/${friendsPost}`)).status, 404);
  assert.equal((await carol.call('POST', `/posts/${friendsPost}/comments`, { body: 'היי' })).status, 404);
  assert.equal((await anon('GET', '/feed')).status, 401);
});

test('a public post opens without an account', async () => {
  assert.equal((await anon('GET', `/posts/${publicPost}`)).data.post.body, 'לכולם');
});

test('ending a friendship ends access', async () => {
  const dave = await signUp('dave');
  await dave.call('POST', `/friends/${alice.id}`);
  await alice.call('POST', `/friends/${dave.id}`);
  assert.equal((await dave.call('GET', `/posts/${friendsPost}`)).status, 200);
  await dave.call('DELETE', `/friends/${alice.id}`);
  assert.equal((await dave.call('GET', `/posts/${friendsPost}`)).status, 404);
});

test('all comments return in the order written, links intact', async () => {
  const bodies = ['ראשונה', 'https://example.com/a', 'שלישית'];
  for (const [i, body] of bodies.entries()) {
    await [bob, carol, bob][i].call('POST', `/posts/${publicPost}/comments`, { body });
  }
  const { comments } = (await anon('GET', `/posts/${publicPost}`)).data;
  assert.deepEqual(comments.map((item) => item.body), bodies);
});

test('a removed comment keeps its place and says who removed it', async () => {
  const before = (await alice.call('GET', `/posts/${publicPost}`)).data.comments;
  const [first, second] = before;
  assert.equal((await carol.call('DELETE', `/comments/${first.id}`)).status, 403);
  assert.equal((await alice.call('DELETE', `/comments/${first.id}`)).status, 200);
  assert.equal((await carol.call('DELETE', `/comments/${second.id}`)).status, 200);
  const after = (await anon('GET', `/posts/${publicPost}`)).data.comments;
  assert.deepEqual(after.map((item) => item.id), before.map((item) => item.id));
  assert.deepEqual(after.map((item) => item.removedBy), ['post_owner', 'author', null]);
  assert.deepEqual(after.map((item) => item.body), [null, null, 'שלישית']);
});

test('a YouTube link in the text becomes a video', async () => {
  const urls = ['https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3', 'https://youtu.be/dQw4w9WgXcQ', 'https://youtube.com/shorts/dQw4w9WgXcQ'];
  for (const url of urls) {
    const { id } = (await alice.call('POST', '/posts', { body: `ראו ${url}`, audience: 'public' })).data;
    assert.equal((await anon('GET', `/posts/${id}`)).data.post.youtubeId, 'dQw4w9WgXcQ');
  }
  const { id } = (await alice.call('POST', '/posts', { body: 'https://evil.example/youtube.com/watch?v=dQw4w9WgXcQ', audience: 'public' })).data;
  assert.equal((await anon('GET', `/posts/${id}`)).data.post.youtubeId, null);
});

test('a YouTube link in a comment becomes a video, until the comment is removed', async () => {
  const { id } = (await bob.call('POST', `/posts/${publicPost}/comments`, { body: 'https://youtu.be/dQw4w9WgXcQ' })).data;
  const find = async () => (await anon('GET', `/posts/${publicPost}`)).data.comments.find((item) => item.id === id);
  assert.equal((await find()).youtubeId, 'dQw4w9WgXcQ');
  await bob.call('DELETE', `/comments/${id}`);
  assert.equal((await find()).youtubeId, null);
});

test('only the author can delete a post, and it disappears', async () => {
  const { id } = (await alice.call('POST', '/posts', { body: 'זמני', audience: 'public' })).data;
  assert.equal((await bob.call('DELETE', `/posts/${id}`)).status, 404);
  assert.equal((await alice.call('DELETE', `/posts/${id}`)).status, 200);
  assert.equal((await anon('GET', `/posts/${id}`)).status, 404);
});

test('only the author rewrites a post, and readers are told it was edited', async () => {
  const { id } = (await alice.call('POST', '/posts', { body: 'נוסח ראשון', audience: 'friends' })).data;
  const read = async (who = bob) => (await who.call('GET', `/posts/${id}`)).data.post;
  assert.equal((await read()).editedAt, null);
  assert.equal((await bob.call('PUT', `/posts/${id}`, { body: 'של מישהו אחר' })).status, 404);
  assert.equal((await anon('PUT', `/posts/${id}`, { body: 'בלי חשבון' })).status, 401);
  // Saving the same words is not an edit.
  assert.equal((await alice.call('PUT', `/posts/${id}`, { body: ' נוסח ראשון ' })).data.post.editedAt, null);

  const started = Date.now();
  const { status, data } = await alice.call('PUT', `/posts/${id}`, { body: 'נוסח שני https://youtu.be/dQw4w9WgXcQ', audience: 'public' });
  assert.equal(status, 200);
  assert.equal(data.post.mine, true);
  const after = await read();
  assert.equal(after.body, 'נוסח שני https://youtu.be/dQw4w9WgXcQ');
  assert.equal(after.youtubeId, 'dQw4w9WgXcQ');
  assert.ok(after.editedAt >= started);
  // Only the words change: the audience stays as the people who answered knew it.
  assert.equal(after.audience, 'friends');
  assert.equal((await carol.call('GET', `/posts/${id}`)).status, 404);
  assert.equal((await bob.call('GET', '/feed')).data.posts.find((post) => post.id === id).body, after.body);

  assert.equal((await alice.call('PUT', `/posts/${id}`, { body: '  ' })).status, 400);
  assert.equal((await alice.call('PUT', `/posts/${id}`, {})).status, 400);
  assert.equal((await alice.call('PUT', `/posts/${id}`, { body: 'א'.repeat(50001) })).status, 400);
  assert.equal((await read()).body, after.body);
  await alice.call('DELETE', `/posts/${id}`);
  assert.equal((await alice.call('PUT', `/posts/${id}`, { body: 'אחרי מחיקה' })).status, 404);
});

test('a post with a picture can lose its words, and keeps the picture', async () => {
  const image = (await alice.call('POST', '/images', phoneJpeg())).data.id;
  const { id } = (await alice.call('POST', '/posts', { body: 'עם תמונה', audience: 'public', images: [image] })).data;
  const { data } = await alice.call('PUT', `/posts/${id}`, { body: '' });
  assert.equal(data.post.body, '');
  assert.deepEqual(data.post.images, [image]);
  assert.ok(data.post.editedAt);
});

test('only the writer rewrites a comment, not the owner of the post, and not after it was removed', async () => {
  const { id } = (await bob.call('POST', `/posts/${publicPost}/comments`, { body: 'תגובה עם טעות' })).data;
  const find = async () => (await anon('GET', `/posts/${publicPost}`)).data.comments.find((item) => item.id === id);
  assert.equal((await find()).editedAt, null);
  const seenBy = async (who) => (await who.call('GET', `/posts/${publicPost}`)).data.comments.find((item) => item.id === id);
  assert.equal((await seenBy(bob)).canEdit, true);
  assert.equal((await seenBy(alice)).canEdit, false);
  assert.equal((await seenBy(alice)).canRemove, true);
  assert.equal((await alice.call('PUT', `/comments/${id}`, { body: 'בעלת הפוסט משכתבת' })).status, 404);
  assert.equal((await anon('PUT', `/comments/${id}`, { body: 'אורח' })).status, 401);
  assert.equal((await bob.call('PUT', `/comments/${id}`, { body: 'תגובה עם טעות' })).status, 200);
  assert.equal((await find()).editedAt, null);

  const started = Date.now();
  assert.equal((await bob.call('PUT', `/comments/${id}`, { body: 'תגובה מתוקנת https://youtu.be/dQw4w9WgXcQ' })).status, 200);
  const after = await find();
  assert.equal(after.body, 'תגובה מתוקנת https://youtu.be/dQw4w9WgXcQ');
  assert.equal(after.youtubeId, 'dQw4w9WgXcQ');
  assert.ok(after.editedAt >= started);
  assert.equal((await bob.call('PUT', `/comments/${id}`, { body: '' })).status, 400);
  assert.equal((await bob.call('PUT', `/comments/${id}`, { body: 'א'.repeat(50001) })).status, 400);

  await bob.call('DELETE', `/comments/${id}`);
  assert.equal((await bob.call('PUT', `/comments/${id}`, { body: 'אחרי הסרה' })).status, 404);
  const removed = await find();
  assert.equal(removed.body, null);
  assert.equal(removed.editedAt, null);
  assert.equal((await seenBy(bob)).canEdit, false);
});

// A tiny JPEG shaped like a phone photo: an Exif block holding a location, then the picture data.
function phoneJpeg() {
  const exif = Buffer.from('Exif\0\0GPS-32.47,34.97');
  return Buffer.concat([
    Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46]),
    Buffer.from([0xff, 0xe1, (exif.length + 2) >> 8, (exif.length + 2) & 255]), exif,
    Buffer.from([0xff, 0xdb, 0x00, 0x04, 0x01, 0x02]),
    Buffer.from([0xff, 0xda, 0x00, 0x02, 0x11, 0x22, 0x33, 0xff, 0xd9]),
  ]);
}

const upload = (user, data) => user.call('POST', '/images', data);
const fetchImage = (user, id) => user('GET', `/images/${id}`);

const kinds = (reactions) => reactions.map((item) => `${item.kind}:${item.count}`);

test('a feeling is one per person: choosing again replaces it, and it can be taken back', async () => {
  const { id } = (await alice.call('POST', '/posts', { body: 'מה דעתכם?', audience: 'friends' })).data;
  assert.deepEqual((await bob.call('PUT', `/posts/${id}/reaction`, { kind: 'love' })).data.reactions,
    [{ kind: 'love', count: 1, mine: true, people: [{ id: bob.id, name: 'bob' }] }]);
  await alice.call('PUT', `/posts/${id}/reaction`, { kind: 'funny' });
  await alice.call('PUT', `/posts/${id}/reaction`, { kind: 'love' });
  // The feed carries the same thing, and tells each reader which feeling is theirs.
  const inFeed = (await alice.call('GET', '/feed')).data.posts.find((post) => post.id === id).reactions;
  assert.deepEqual(inFeed, [{ kind: 'love', count: 2, mine: true, people: [{ id: bob.id, name: 'bob' }, { id: alice.id, name: 'alice' }] }]);

  assert.deepEqual(kinds((await bob.call('PUT', `/posts/${id}/reaction`, { kind: 'anger' })).data.reactions), ['love:1', 'anger:1']);
  assert.deepEqual(kinds((await bob.call('DELETE', `/posts/${id}/reaction`)).data.reactions), ['love:1']);
  assert.deepEqual(kinds((await bob.call('DELETE', `/posts/${id}/reaction`)).data.reactions), ['love:1']);
  const seen = (await bob.call('GET', `/users/${alice.id}`)).data.posts.find((post) => post.id === id).reactions;
  assert.deepEqual(seen.map((item) => item.mine), [false]);
});

test('a feeling follows the audience of the post, and a visitor sees counts without names', async () => {
  const hidden = (await alice.call('POST', '/posts', { body: 'לחברים', audience: 'friends' })).data.id;
  const open = (await alice.call('POST', '/posts', { body: 'לכולם', audience: 'public' })).data.id;
  assert.equal((await carol.call('PUT', `/posts/${hidden}/reaction`, { kind: 'love' })).status, 404);
  assert.equal((await carol.call('DELETE', `/posts/${hidden}/reaction`)).status, 404);
  assert.equal((await anon('PUT', `/posts/${open}/reaction`, { kind: 'love' })).status, 401);
  assert.equal((await carol.call('PUT', `/posts/${open}/reaction`, { kind: 'surprise' })).status, 200);
  assert.deepEqual((await anon('GET', `/posts/${open}`)).data.post.reactions, [{ kind: 'surprise', count: 1, mine: false, people: [] }]);
});

test('any single emoji is a feeling, and nothing else is', async () => {
  const { id } = (await alice.call('POST', '/posts', { body: 'אימוג׳י', audience: 'friends' })).data;
  for (const kind of ['🌻', '👍🏽', '🇮🇱', '👨‍👩‍👧', '🫶']) {
    assert.deepEqual(kinds((await bob.call('PUT', `/posts/${id}/reaction`, { kind })).data.reactions), [`${kind}:1`]);
  }
  for (const kind of ['', 'like', 'a', '7', '🌻🌻', '🌻 ', '<b>', 'אהבתי', null, 5, ['love'], 'x'.repeat(500)]) {
    assert.equal((await bob.call('PUT', `/posts/${id}/reaction`, { kind })).status, 400, String(kind));
  }
  assert.equal((await bob.call('PUT', `/posts/${id}/reaction`)).status, 400);
  // An emoji that says what a drawn feeling says is counted with it.
  await alice.call('PUT', `/posts/${id}/reaction`, { kind: 'funny' });
  assert.deepEqual(kinds((await bob.call('PUT', `/posts/${id}/reaction`, { kind: '😂' })).data.reactions), ['funny:2']);
  assert.deepEqual(kinds((await bob.call('PUT', `/posts/${id}/reaction`, { kind: '❤️' })).data.reactions), ['funny:1', 'love:1']);
});

test('a comment gathers feelings of its own, until it is removed or its post is deleted', async () => {
  const { id } = (await alice.call('POST', '/posts', { body: 'פוסט עם תגובות', audience: 'friends' })).data;
  const first = (await bob.call('POST', `/posts/${id}/comments`, { body: 'תגובה' })).data.id;
  const second = (await alice.call('POST', `/posts/${id}/comments`, { body: 'עוד אחת' })).data.id;
  assert.deepEqual(kinds((await alice.call('PUT', `/comments/${first}/reaction`, { kind: 'bored' })).data.reactions), ['bored:1']);
  await bob.call('PUT', `/comments/${second}/reaction`, { kind: '🌻' });
  await bob.call('PUT', `/posts/${id}/reaction`, { kind: 'love' });
  assert.equal((await carol.call('PUT', `/comments/${first}/reaction`, { kind: 'love' })).status, 404);
  assert.equal((await alice.call('PUT', '/comments/999999/reaction', { kind: 'love' })).status, 404);
  const read = async () => (await bob.call('GET', `/posts/${id}`)).data;
  assert.deepEqual((await read()).comments.map((item) => kinds(item.reactions)), [['bored:1'], ['🌻:1']]);
  assert.deepEqual((await read()).comments[1].reactions[0].mine, true);
  assert.deepEqual(kinds((await read()).post.reactions), ['love:1']);

  // A removed comment loses what was felt about it, and takes no more.
  await bob.call('DELETE', `/comments/${first}`);
  assert.deepEqual((await read()).comments.map((item) => kinds(item.reactions)), [[], ['🌻:1']]);
  assert.equal((await alice.call('PUT', `/comments/${first}/reaction`, { kind: 'love' })).status, 404);

  await alice.call('DELETE', `/posts/${id}`);
  assert.equal((await bob.call('PUT', `/posts/${id}/reaction`, { kind: 'love' })).status, 404);
  assert.equal((await bob.call('PUT', `/comments/${second}/reaction`, { kind: 'love' })).status, 404);
});

test('a share quotes the original post, only for whoever may read it', async () => {
  const writer = await signUp('share-writer');
  const friend = await signUp('share-friend');
  const far = await signUp('share-far');
  for (const [a, b] of [[writer, friend], [friend, far]]) {
    await a.call('POST', `/friends/${b.id}`);
    await b.call('POST', `/friends/${a.id}`);
  }
  const closed = (await writer.call('POST', '/posts', { body: 'רק לחברים', audience: 'friends' })).data.id;

  // Nobody shares their own post, or a post they cannot read.
  assert.equal((await writer.call('POST', `/posts/${closed}/share`)).status, 400);
  assert.equal((await far.call('POST', `/posts/${closed}/share`)).status, 404);
  assert.equal((await anon('POST', `/posts/${closed}/share`)).status, 401);

  // One click, once: sharing again changes nothing.
  const first = await friend.call('POST', `/posts/${closed}/share`);
  assert.equal(first.status, 201);
  const share = first.data.id;
  assert.deepEqual(first.data.shares, { count: 1, mine: share });
  assert.equal((await friend.call('POST', `/posts/${closed}/share`)).data.id, share);

  // The share is a post of the friend, for the friend's friends, and it carries the original.
  const mine = (await friend.call('GET', '/feed')).data.posts.find((post) => post.id === share);
  assert.deepEqual([mine.author.id, mine.body, mine.audience, mine.shares], [friend.id, '', 'friends', null]);
  assert.deepEqual([mine.shared.post.id, mine.shared.post.body, mine.shared.post.author.id], [closed, 'רק לחברים', writer.id]);
  assert.deepEqual(mine.shared.post.shares, { count: 1, mine: share });
  assert.deepEqual((await writer.call('GET', `/posts/${closed}`)).data.post.shares, { count: 1, mine: null });

  // A friend of the friend, who is not a friend of the writer, sees the share without the words.
  const seen = (await far.call('GET', '/feed')).data.posts.find((post) => post.id === share);
  assert.deepEqual(seen.shared, { post: null, gone: false });
  assert.deepEqual((await far.call('GET', `/posts/${share}`)).data.post.shared, { post: null, gone: false });
  assert.equal((await far.call('POST', `/posts/${share}/share`)).status, 404);
  assert.equal((await anon('GET', `/posts/${share}`)).status, 404);

  // Taking the share back frees the post to be shared again.
  assert.equal((await friend.call('DELETE', `/posts/${share}`)).status, 200);
  assert.deepEqual((await writer.call('GET', `/posts/${closed}`)).data.post.shares, { count: 0, mine: null });
  const again = (await friend.call('POST', `/posts/${closed}/share`)).data.id;
  assert.notEqual(again, share);
});

test('the next post starts from the audience chosen for the last one', async () => {
  const writer = await signUp('audience-writer');
  const friend = await signUp('audience-friend');
  await writer.call('POST', `/friends/${friend.id}`);
  await friend.call('POST', `/friends/${writer.id}`);
  const last = async (who) => (await who.call('GET', '/audience')).data.last;
  assert.equal(await last(writer), 'friends');
  const open = (await writer.call('POST', '/posts', { body: 'לכולם', audience: 'public' })).data.id;
  assert.equal(await last(writer), 'public');
  assert.equal(await last(friend), 'friends');

  // A share goes to friends without asking, so it does not change the choice.
  await friend.call('POST', '/posts', { body: 'גם אני לכולם', audience: 'public' });
  await friend.call('POST', `/posts/${open}/share`);
  assert.equal(await last(friend), 'public');

  // The choice holds after the post is deleted, and follows the next post.
  await writer.call('DELETE', `/posts/${open}`);
  assert.equal(await last(writer), 'public');
  await writer.call('POST', '/posts', { body: 'רק לחברים', audience: 'friends' });
  assert.equal(await last(writer), 'friends');
});

test('the writer of a post sees exactly who shared it, also through another share', async () => {
  const writer = await signUp('echo-writer');
  const friend = await signUp('echo-friend');
  const far = await signUp('echo-far');
  for (const [a, b] of [[writer, friend], [friend, far]]) {
    await a.call('POST', `/friends/${b.id}`);
    await b.call('POST', `/friends/${a.id}`);
  }
  const open = (await writer.call('POST', '/posts', { body: 'לכל העולם', audience: 'public' })).data.id;
  const near = (await friend.call('POST', `/posts/${open}/share`)).data.id;

  // Sharing from the friend's share is a share of the original post.
  const through = await far.call('POST', `/posts/${near}/share`);
  assert.deepEqual(through.data.shares, { count: 2, mine: through.data.id });
  const quote = (await far.call('GET', `/posts/${through.data.id}`)).data.post.shared.post;
  assert.deepEqual([quote.id, quote.author.id], [open, writer.id]);
  assert.equal((await anon('GET', `/posts/${open}`)).data.post.shares.count, 2);

  // The writer gets the names, the newest first, and opens only the share of a friend.
  const { shares } = (await writer.call('GET', `/posts/${open}/shares`)).data;
  assert.deepEqual(shares.map((item) => [item.user.id, item.postId, item.via?.id ?? null]),
    [[far.id, null, friend.id], [friend.id, near, null]]);
  assert.equal((await friend.call('GET', `/posts/${open}/shares`)).status, 404);
  assert.equal((await writer.call('GET', `/posts/${near}/shares`)).status, 404);
  const news = (await writer.call('GET', '/update')).data.friends.shared;
  assert.deepEqual(news.map((item) => [item.user.id, item.postId, item.shareId]), [[far.id, open, null], [friend.id, open, near]]);
  assert.deepEqual((await friend.call('GET', '/update')).data.friends.shared, []);

  // Words can be added to a share and taken away again. A post of one's own cannot be emptied.
  const worded = await friend.call('PUT', `/posts/${near}`, { body: 'כדאי לקרוא' });
  assert.deepEqual([worded.data.post.body, worded.data.post.shared.post.id], ['כדאי לקרוא', open]);
  assert.equal((await far.call('PUT', `/posts/${through.data.id}`, { body: '' })).status, 200);
  assert.equal((await writer.call('PUT', `/posts/${open}`, { body: '' })).status, 400);

  // When the original is deleted, a share that says something stays and says so.
  // A share that was only the quote goes with it, with what was written under it.
  await writer.call('POST', `/posts/${through.data.id}/comments`, { body: 'תודה' });
  await writer.call('PUT', `/posts/${through.data.id}/reaction`, { kind: 'love' });
  assert.equal((await writer.call('DELETE', `/posts/${open}`)).status, 200);
  assert.deepEqual((await friend.call('GET', `/posts/${near}`)).data.post.shared, { post: null, gone: true });
  assert.equal((await far.call('GET', `/posts/${through.data.id}`)).status, 404);
  assert.equal((await friend.call('POST', `/posts/${near}/share`)).status, 404);
  assert.deepEqual((await writer.call('GET', '/update')).data.friends.shared, []);
});

test('location data never survives an upload', async () => {
  const { status, data } = await upload(alice, phoneJpeg());
  assert.equal(status, 201);
  const { id } = (await alice.call('POST', '/posts', { body: '', audience: 'public', images: [data.id] })).data;
  assert.deepEqual((await anon('GET', `/posts/${id}`)).data.post.images, [data.id]);
  const stored = Buffer.from((await anon('GET', `/images/${data.id}`)).bytes);
  assert.ok(!stored.includes('GPS'));
  assert.ok(stored.includes(Buffer.from([0xff, 0xda, 0x00, 0x02, 0x11, 0x22, 0x33, 0xff, 0xd9])));
});

test('only JPEG files within the size limit are accepted', async () => {
  assert.equal((await upload(alice, Buffer.from('<svg onload="alert(1)"/>'))).status, 415);
  assert.equal((await upload(alice, Buffer.alloc(1024 * 1024 + 1))).status, 413);
});

test('an image follows the audience of its post', async () => {
  const image = (await upload(alice, phoneJpeg())).data.id;
  assert.equal((await fetchImage(alice.call, image)).status, 200);
  assert.equal((await fetchImage(bob.call, image)).status, 404);
  await alice.call('POST', '/posts', { body: 'פרטי', audience: 'friends', images: [image] });
  assert.equal((await fetchImage(bob.call, image)).status, 200);
  assert.equal((await fetchImage(carol.call, image)).status, 404);
  assert.equal((await fetchImage(anon, image)).status, 404);
});

test('an image cannot be attached twice or by someone else', async () => {
  const image = (await upload(alice, phoneJpeg())).data.id;
  assert.equal((await bob.call('POST', '/posts', { body: 'x', audience: 'public', images: [image] })).status, 400);
  assert.equal((await alice.call('POST', '/posts', { body: 'x', audience: 'friends', images: [image] })).status, 201);
  assert.equal((await alice.call('POST', '/posts', { body: 'y', audience: 'public', images: [image] })).status, 400);
});

test('deleting a post deletes its images', async () => {
  const image = (await upload(alice, phoneJpeg())).data.id;
  const { id } = (await alice.call('POST', '/posts', { body: 'x', audience: 'public', images: [image] })).data;
  assert.equal((await fetchImage(anon, image)).status, 200);
  await alice.call('DELETE', `/posts/${id}`);
  assert.equal((await fetchImage(alice.call, image)).status, 404);
});

test('a post needs text or an image, and at most four images', async () => {
  assert.equal((await alice.call('POST', '/posts', { body: '  ', audience: 'public' })).status, 400);
  const images = [];
  for (let i = 0; i < 5; i++) images.push((await upload(alice, phoneJpeg())).data.id);
  assert.equal((await alice.call('POST', '/posts', { body: 'x', audience: 'public', images })).status, 400);
});

const site = (name) => `http://localhost:${YOUTUBE_PORT}/site/${name}`;

test('a live page is read once for its title and picture, and follows the audience of its post', async () => {
  const { data, status } = await alice.call('POST', '/posts', { body: '', audience: 'friends', page: site('lab') });
  assert.equal(status, 201);
  const { post } = (await alice.call('GET', `/posts/${data.id}`)).data;
  assert.deepEqual(post.page, { url: site('lab'), title: 'כותרת & משנה', text: 'מעבדה שאפשר לשחק בה', picture: `/api/posts/${data.id}/page-picture`, framable: true });
  const picture = await bob.call('GET', `/posts/${data.id}/page-picture`);
  assert.equal(picture.status, 200);
  assert.equal(Buffer.from(picture.bytes)[1], 0x50);
  assert.equal((await anon('GET', `/posts/${data.id}/page-picture`)).status, 404);
  assert.equal((await anon('GET', `/posts/${data.id}`)).status, 404);
});

test('a live page that refuses frames is marked so, unless its policy lets anyone frame it', async () => {
  for (const [name, framable] of [['denied', false], ['own-only', false], ['anyone', true]]) {
    const { data } = await alice.call('POST', '/posts', { body: name, audience: 'public', page: site(name) });
    assert.equal((await anon('GET', `/posts/${data.id}`)).data.post.page.framable, framable, name);
  }
});

test('only a full outside address can be a live page', async () => {
  for (const page of ['javascript:alert(1)', 'data:text/html,<b>x</b>', 'ftp://example.com/', 'not a url', 'https://user:pass@example.com/',
    `http://localhost:${PORT}/`, `http://localhost:${PORT}/api/me`]) {
    assert.equal((await alice.call('POST', '/posts', { body: 'x', audience: 'public', page })).status, 400, page);
  }
  const { data } = await alice.call('POST', '/posts', { body: 'בלי דף', audience: 'public', page: '  ' });
  assert.equal((await anon('GET', `/posts/${data.id}`)).data.post.page, null);
});

test('the writer can add, change and take away the live page, and readers see it was edited', async () => {
  const { data } = await alice.call('POST', '/posts', { body: 'דף', audience: 'public' });
  const put = (body) => alice.call('PUT', `/posts/${data.id}`, body);
  let { post } = (await put({ body: 'דף', page: site('lab') })).data;
  assert.equal(post.page.title, 'כותרת & משנה');
  assert.ok(post.editedAt);
  ({ post } = (await put({ body: 'דף אחר' })).data);
  assert.equal(post.page.url, site('lab'));
  ({ post } = (await put({ body: 'דף אחר', page: site('denied') })).data);
  assert.equal(post.page.framable, false);
  ({ post } = (await put({ body: 'דף אחר', page: '' })).data);
  assert.equal(post.page, null);
  assert.equal((await bob.call('PUT', `/posts/${data.id}`, { body: 'x', page: site('lab') })).status, 404);
  // A post that is only its page cannot lose both.
  const only = (await alice.call('POST', '/posts', { body: '', audience: 'public', page: site('lab') })).data;
  assert.equal((await alice.call('PUT', `/posts/${only.id}`, { body: '', page: '' })).status, 400);
});

test('name search treats wildcards literally', async () => {
  assert.equal((await bob.call('GET', '/users?q=al')).data.users.length, 1);
  assert.equal((await bob.call('GET', '/users?q=%25%25')).data.users.length, 0);
});

test('a crash reaches the log with a code the user can quote', async () => {
  const state = join(root, String(PROD_PORT));
  const wrangler = ['wrangler', 'd1', 'execute', 'free-network', '--local', '--persist-to', state, '--command'];
  execFileSync('npx', [...wrangler, 'DROP TABLE images'], { stdio: 'ignore', env: { ...process.env, CI: '1' } });
  const { status, data } = await client(PROD_PORT)('GET', '/posts/1');
  assert.equal(status, 500);
  assert.match(data.requestId, /^[0-9a-f]{8}$/);
  assert.ok(!JSON.stringify(data).includes('images'));
  const log = execFileSync('node', ['tools/logs.mjs', '--persist-to', state, '--id', data.requestId], { encoding: 'utf8' });
  assert.match(log, /no such table: images/);
  assert.match(log, /GET \/api\/posts\/1 500/);
});

test('the journal tells who got in where and how, and keeps nothing a person wrote', async () => {
  const state = join(root, String(PORT));
  const read = (...more) => execFileSync('node', ['tools/logs.mjs', '--persist-to', state, ...more], { encoding: 'utf8' });
  const guy = await signUp('journal-guy');
  const first = guy.call.cookie();
  // The browser says how the network was opened. Words outside the short list of letters are dropped.
  await guy.call('POST', '/journal', { mode: 'app', display: 'standalone', from: 'il.pardes.freenetwork', path: '/p/N', app: '1.1', note: 'שלום <script>', extra: 'x' });
  await client()('POST', '/journal', { mode: 'browser', display: 'tab', path: '/login' });
  // Logging in again in the same browser ends the session it held.
  const { data } = await guy.call('POST', '/auth/request', { email: 'journal-guy@example.com' });
  assert.equal((await guy.call('POST', '/auth/verify', { token: data.devLink.split('#token=')[1] })).status, 200);
  const stale = await fetch(`http://localhost:${PORT}/api/me`, { headers: { Cookie: first } });
  assert.equal((await stale.json()).user, null);
  assert.equal((await guy.call('GET', '/me')).data.user.id, guy.id);

  const lines = read('--journal', '--user', String(guy.id));
  assert.match(lines, /login +member \d+ journal-guy/);
  assert.match(lines, /link, joined/);
  assert.match(lines, /mode=app display=standalone from=il\.pardes\.freenetwork path=\/p\/N app=1\.1\n/);
  assert.match(lines, /link, replacing their older session here/);
  assert.ok(!lines.includes('script') && !lines.includes('extra'));
  assert.match(read('--journal', '--kind', 'open', '--limit', '1'), /nobody logged in.*\n {4}mode=browser display=tab path=\/login/);

  const sessions = read('--sessions', '--user', String(guy.id));
  assert.match(sessions, /1 sessions/);
  assert.match(sessions, /began 20\d\d-.* last used 20\d\d-/);
  await guy.call('POST', '/auth/logout');
  assert.match(read('--journal', '--user', String(guy.id), '--kind', 'logout'), /1 lines/);
  assert.match(read('--sessions', '--user', String(guy.id)), /0 sessions/);
});

test('the journal names the device in a few words', () => {
  assert.equal(deviceName('Mozilla/5.0 (Linux; Android 14; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36'), 'Android 14 · Chrome 131');
  assert.equal(deviceName('Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1'), 'iOS 17.5 · Safari 17');
  assert.equal(deviceName('Mozilla/5.0 (Linux; Android 13; Pixel 7 Build/TQ3A; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/120.0.0.0 Mobile Safari/537.36'), 'Android 13 · Chrome 120 (webview)');
  assert.equal(deviceName('Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0'), 'Windows · Firefox 132');
  assert.equal(deviceName(undefined), null);
});

test('browser errors are logged, and post text is not', async () => {
  const state = join(root, String(PORT));
  await client()('POST', '/client-log', { message: 'TypeError: boom', stack: 'at app.js:1', path: '/p/1' });
  await alice.call('POST', '/posts', { body: 'טקסט סודי', audience: 'nobody' });
  const log = execFileSync('node', ['tools/logs.mjs', '--persist-to', state], { encoding: 'utf8' });
  assert.match(log, /TypeError: boom/);
  assert.match(log, /POST \/api\/posts 400/);
  assert.ok(!log.includes('טקסט סודי'));
});

test('a comment can be an image, and it follows the audience of the post', async () => {
  const noa = await signUp('noa');
  const yael = await signUp('yael');
  const outsider = await signUp('outsider');
  await noa.call('POST', `/friends/${yael.id}`);
  await yael.call('POST', `/friends/${noa.id}`);
  const { data: { id: postId } } = await noa.call('POST', '/posts', { body: 'לחברים', audience: 'friends' });
  const image = (await upload(yael, phoneJpeg())).data.id;
  assert.equal((await yael.call('POST', `/posts/${postId}/comments`, { image })).status, 201);
  const { data } = await noa.call('GET', `/posts/${postId}`);
  assert.equal(data.comments[0].image, image);
  assert.equal(data.comments[0].body, '');
  assert.equal((await fetchImage(noa.call, image)).status, 200);
  assert.equal((await fetchImage(outsider.call, image)).status, 404);
  assert.equal((await fetchImage(anon, image)).status, 404);
});

test('a comment needs text or an image, and an image is used once', async () => {
  const roni = await signUp('roni');
  const { data: { id: postId } } = await roni.call('POST', '/posts', { body: 'x', audience: 'public' });
  assert.equal((await roni.call('POST', `/posts/${postId}/comments`, { body: '  ' })).status, 400);
  const image = (await upload(roni, phoneJpeg())).data.id;
  assert.equal((await alice.call('POST', `/posts/${postId}/comments`, { image })).status, 400);
  assert.equal((await roni.call('POST', `/posts/${postId}/comments`, { image })).status, 201);
  assert.equal((await roni.call('POST', `/posts/${postId}/comments`, { image })).status, 400);
  assert.equal((await roni.call('POST', '/posts', { body: '', audience: 'public', images: [image] })).status, 400);
});

test('removing a comment or its post really deletes the image', async () => {
  const tal = await signUp('tal');
  const { data: { id: postId } } = await tal.call('POST', '/posts', { body: 'x', audience: 'public' });
  const first = (await upload(tal, phoneJpeg())).data.id;
  const second = (await upload(tal, phoneJpeg())).data.id;
  const { data: { id: commentId } } = await tal.call('POST', `/posts/${postId}/comments`, { image: first });
  await tal.call('POST', `/posts/${postId}/comments`, { body: 'עם תמונה', image: second });
  await tal.call('DELETE', `/comments/${commentId}`);
  assert.equal((await fetchImage(tal.call, first)).status, 404);
  assert.equal((await tal.call('GET', `/posts/${postId}`)).data.comments[0].image, null);
  assert.equal((await fetchImage(anon, second)).status, 200);
  await tal.call('DELETE', `/posts/${postId}`);
  assert.equal((await fetchImage(tal.call, second)).status, 404);
});

test('a stranger cannot flood the journal or the log, and what is written is not kept by anyone on the way', async () => {
  const state = join(root, String(UPLOADS_PORT));
  const count = (...more) => Number(execFileSync('node', ['tools/logs.mjs', '--persist-to', state, '--limit', '1000', ...more], { encoding: 'utf8' }).match(/(\d+) (lines|entries)\n$/)[1]);
  const nobody = client(UPLOADS_PORT);
  const member = await signUp('flooded', UPLOADS_PORT);
  for (let round = 0; round < 16; round++) {
    await Promise.all(Array.from({ length: 20 }, () => [
      nobody('POST', '/journal', { mode: 'browser' }),
      nobody('POST', '/client-log', { message: 'flood' }),
      nobody('POST', '/auth/verify', { token: 'no-such-link' }),
    ]).flat());
  }
  assert.equal(count('--journal', '--kind', 'open'), 300);
  // Browser errors and refusals each stop at their own limit.
  assert.equal(count(), 600);
  // A member still gets their own lines, and a refusal answers the same whether it was written or not.
  const opened = await member.call('POST', '/journal', { mode: 'app' });
  assert.equal(opened.status, 200);
  assert.equal(count('--journal', '--user', String(member.id), '--kind', 'open'), 1);
  assert.equal((await nobody('POST', '/auth/verify', { token: 'no-such-link' })).status, 400);

  const answer = await fetch(`http://localhost:${UPLOADS_PORT}/api/me`);
  assert.equal(answer.headers.get('cache-control'), 'private, no-store');
});

test('a long text is taken whole, in a post, a comment and a message, up to one clear limit', async () => {
  const [writer, friend] = [await signUp('longwriter'), await signUp('longfriend')];
  await writer.call('POST', `/friends/${friend.id}`);
  await friend.call('POST', `/friends/${writer.id}`);
  const long = 'מילה '.repeat(10000).trim();
  assert.equal(long.length, 49999);
  const { data: { id } } = await writer.call('POST', '/posts', { body: long, audience: 'friends' });
  const { data: { id: commentId } } = await friend.call('POST', `/posts/${id}/comments`, { body: long });
  const read = (await writer.call('GET', `/posts/${id}`)).data;
  assert.equal(read.post.body, long);
  assert.equal(read.comments.find((item) => item.id === commentId).body, long);
  assert.equal((await writer.call('POST', `/messages/${friend.id}`, { body: long })).data.message.body, long);
  const tooLong = `${long}xx`;
  assert.equal((await writer.call('POST', '/posts', { body: tooLong, audience: 'friends' })).status, 400);
  assert.equal((await friend.call('POST', `/posts/${id}/comments`, { body: tooLong })).status, 400);
  assert.equal((await writer.call('POST', `/messages/${friend.id}`, { body: tooLong })).status, 400);
});

test('a member sets their own preferences, and they follow them to another device', async () => {
  const picky = await signUp('picky');
  assert.deepEqual((await picky.call('GET', '/me')).data.user.prefs, { enterPosts: false, enterMessages: true });
  assert.deepEqual((await picky.call('PUT', '/me/prefs', { enterPosts: true })).data.prefs, { enterPosts: true, enterMessages: true });
  assert.deepEqual((await picky.call('PUT', '/me/prefs', { enterMessages: false })).data.prefs, { enterPosts: true, enterMessages: false });
  // Another device of the same member, logged in with a new link.
  const other = client();
  const { data } = await other('POST', '/auth/request', { email: 'picky@example.com' });
  await other('POST', '/auth/verify', { token: data.devLink.split('#token=')[1] });
  assert.deepEqual((await other('GET', '/me')).data.user.prefs, { enterPosts: true, enterMessages: false });
  // Only known preferences, each of its own kind, and nobody else's.
  assert.equal((await picky.call('PUT', '/me/prefs', { enterPosts: 'yes' })).status, 400);
  assert.equal((await picky.call('PUT', '/me/prefs', { admin: true })).status, 400);
  assert.equal((await anon('PUT', '/me/prefs', { enterPosts: true })).status, 401);
  assert.deepEqual((await alice.call('GET', '/me')).data.user.prefs, { enterPosts: false, enterMessages: true });
});

test('a member takes out everything they have here, and nothing that is not theirs to read', async () => {
  const [mover, pal, outsider] = [await signUp('mover'), await signUp('mover-pal'), await signUp('mover-outsider')];
  await mover.call('POST', `/friends/${pal.id}`);
  await pal.call('POST', `/friends/${mover.id}`);
  const { data: { id: picture } } = await mover.call('POST', '/images', phoneJpeg());
  const { data: { id: mine } } = await mover.call('POST', '/posts', { body: 'הפוסט שלי', audience: 'friends', images: [picture] });
  const { data: { id: said } } = await pal.call('POST', `/posts/${mine}/comments`, { body: 'תגובה של חבר' });
  await mover.call('POST', `/posts/${mine}/comments`, { body: 'תשובה שלי', parentId: said });
  await pal.call('PUT', `/posts/${mine}/reaction`, { kind: 'love' });
  const { data: { id: theirs } } = await pal.call('POST', '/posts', { body: 'פוסט של חבר', audience: 'friends' });
  await mover.call('POST', `/posts/${theirs}/comments`, { body: 'כתבתי אצל חבר' });
  await mover.call('PUT', `/posts/${theirs}/reaction`, { kind: 'funny' });
  await outsider.call('POST', '/posts', { body: 'סוד של זר', audience: 'friends' });
  await mover.call('POST', `/messages/${pal.id}`, { body: 'הודעה ממני' });
  await pal.call('POST', `/messages/${mover.id}`, { body: 'הודעה אליי' });
  await pal.call('POST', `/friends/${outsider.id}`);
  await outsider.call('POST', `/friends/${pal.id}`);
  await pal.call('POST', `/messages/${outsider.id}`, { body: 'שיחה של אחרים' });
  await mover.call('PUT', '/me/prefs', { enterPosts: true });

  const { status, data } = await mover.call('GET', '/me/export');
  assert.equal(status, 200);
  assert.equal(data.format, 'free-network-export');
  assert.equal(data.profile.name, 'mover');
  assert.equal(data.profile.prefs.enterPosts, true);
  assert.deepEqual(data.friends.map((friend) => [friend.name, friend.status]), [['mover-pal', 'friends']]);
  assert.equal(data.posts.length, 1);
  assert.deepEqual(data.posts[0].images, [picture]);
  assert.deepEqual(data.posts[0].comments.map((item) => [item.author.name, item.body, item.parentId]), [['mover-pal', 'תגובה של חבר', null], ['mover', 'תשובה שלי', said]]);
  assert.deepEqual(data.posts[0].reactions.map((group) => [group.kind, group.people.map((person) => person.name)]), [['love', ['mover-pal']]]);
  assert.deepEqual(data.comments.map((item) => [item.body, item.post.author.name, item.post.words]), [['כתבתי אצל חבר', 'mover-pal', 'פוסט של חבר']]);
  assert.deepEqual(data.reactions.map((item) => [item.kind, item.postId]), [['funny', theirs]]);
  assert.deepEqual(data.conversations.map((talk) => [talk.with.name, talk.messages.map((message) => [message.mine, message.body])]),
    [['mover-pal', [[true, 'הודעה ממני'], [false, 'הודעה אליי']]]]);
  const all = JSON.stringify(data);
  assert.ok(!all.includes('סוד של זר') && !all.includes('שיחה של אחרים') && !all.includes('example.com'));
  // After the friendship ends, the words of the friend's post are no longer handed over.
  await mover.call('DELETE', `/friends/${pal.id}`);
  assert.equal((await mover.call('GET', '/me/export')).data.comments[0].post.words, null);
  assert.equal((await anon('GET', '/me/export')).status, 401);
});

test('the bot of the network writes with every member, is marked as a bot, and reads no more than a stranger', async () => {
  const [bot, member, other] = [await signUp('helper-bot'), await signUp('bot-reader'), await signUp('bot-other')];
  assert.equal((await bot.call('POST', `/messages/${member.id}`, { body: 'לפני' })).status, 403);
  execFileSync('npx', ['wrangler', 'd1', 'execute', 'free-network', '--local', '--persist-to', join(root, String(PORT)), '--command', `UPDATE users SET bot = 1 WHERE id = ${bot.id}`],
    { stdio: 'ignore', env: { ...process.env, CI: '1' } });
  assert.equal((await bot.call('POST', `/messages/${member.id}`, { body: 'שלום, אני הבוט' })).status, 201);
  const talk = (await member.call('GET', `/messages/${bot.id}`)).data;
  assert.equal(talk.canSend, true);
  assert.equal(talk.user.bot, true);
  assert.deepEqual(talk.messages.map((message) => message.body), ['שלום, אני הבוט']);
  assert.equal((await member.call('GET', '/me')).data.user.unreadMessages, 1);
  assert.equal((await member.call('POST', `/messages/${bot.id}`, { body: 'תודה' })).status, 201);
  assert.equal((await bot.call('GET', '/me')).data.user.unreadMessages, 1);
  assert.equal((await member.call('GET', `/users/${bot.id}`)).data.user.bot, true);
  assert.equal((await member.call('GET', `/users/${other.id}`)).data.user.bot, false);
  // Two members who are not friends still cannot write to each other, and the bot is no friend:
  // a post for friends stays closed to it.
  assert.equal((await member.call('POST', `/messages/${other.id}`, { body: 'זר' })).status, 403);
  const { data: { id } } = await member.call('POST', '/posts', { body: 'רק לחברים', audience: 'friends' });
  assert.equal((await bot.call('GET', `/posts/${id}`)).status, 404);
  assert.equal((await bot.call('GET', '/feed')).data.posts.length, 0);
});

test('a reply to a reply stays at the second level, and replies cannot cross posts', async () => {
  const omer = await signUp('omer');
  const { data: { id: postId } } = await omer.call('POST', '/posts', { body: 'x', audience: 'public' });
  const { data: { id: otherPost } } = await omer.call('POST', '/posts', { body: 'y', audience: 'public' });
  const send = (id, body, parentId) => omer.call('POST', `/posts/${id}/comments`, { body, parentId });
  const top = (await send(postId, 'ראשונה')).data.id;
  const reply = (await send(postId, 'תשובה', top)).data.id;
  await send(postId, 'תשובה לתשובה', reply);
  const { comments } = (await anon('GET', `/posts/${postId}`)).data;
  assert.deepEqual(comments.map((item) => item.parentId), [null, top, top]);
  assert.equal((await send(otherPost, 'זר', top)).status, 404);
  assert.equal((await send(postId, 'משובש', { id: top })).status, 404);
  await omer.call('DELETE', `/comments/${reply}`);
  assert.equal((await send(postId, 'להוסרה', reply)).status, 404);
});

test('a video title and picture come through the server, without the reader', async () => {
  const dan = await signUp('dan');
  assert.equal((await dan.call('GET', `/youtube/${VIDEO_ID}`)).data.title, 'סרטון לדוגמה');
  const picture = await dan.call('GET', `/youtube/${VIDEO_ID}/thumb`);
  assert.equal(picture.status, 200);
  assert.equal(Buffer.from(picture.bytes).length, 4);
  assert.equal(youtubeRequests.length, 2);
  for (const { headers } of youtubeRequests) assert.equal(headers.cookie, undefined);
});

test('aerial pictures of the globe come through the server, for members only', async () => {
  const tile = await alice.call('GET', '/tiles/12/2445/1664');
  assert.equal(tile.status, 200);
  assert.equal(Buffer.from(tile.bytes).length, 4);
  assert.equal(tileRequests.length, 1);
  assert.equal(tileRequests[0].headers.cookie, undefined);
  assert.equal((await anon('GET', '/tiles/12/2445/1664')).status, 401);
  // What the source sends that is not a picture is never passed on.
  assert.equal((await alice.call('GET', '/tiles/12/2446/1664')).status, 502);
  assert.equal((await alice.call('GET', '/tiles/12/1/1')).status, 502);
  // Only real tiles are asked for: within the world, and within the zooms that have pictures.
  const asked = tileRequests.length;
  for (const path of ['2/1/1', '18/1/1', '12/4096/0', '12/0/4096', '12/-1/0', '12/a/0', '12/1.5/0', '12/0/0/0']) {
    assert.equal((await alice.call('GET', `/tiles/${path}`)).status, 404, path);
  }
  assert.equal(tileRequests.length, asked);
});

test('a missing video falls back quietly, and odd addresses are refused', async () => {
  assert.equal((await anon('GET', '/youtube/zzzzzzzzzzz')).data.title, null);
  assert.equal((await anon('GET', '/youtube/zzzzzzzzzzz/thumb')).status, 404);
  assert.equal((await anon('GET', '/youtube/short')).status, 404);
  assert.equal((await anon('GET', '/youtube/..%2f..%2fetc00/thumb')).status, 404);
});

const setPicture = (user, kind, imageId) => user.call('PUT', '/me/images', { kind, imageId });

test('a profile picture appears beside posts and comments, even for visitors', async () => {
  const dana = await signUp('dana');
  const { data: { id } } = await upload(dana, phoneJpeg());
  assert.equal((await setPicture(dana, 'avatar', id)).status, 200);
  const { data: post } = await dana.call('POST', '/posts', { body: 'שלום', audience: 'public' });
  await dana.call('POST', `/posts/${post.id}/comments`, { body: 'תגובה' });
  const { data } = await anon('GET', `/posts/${post.id}`);
  assert.equal(data.post.author.avatar, id);
  assert.equal(data.comments[0].author.avatar, id);
  assert.equal((await fetchImage(anon, id)).status, 200);
});

test('replacing or removing a picture deletes the old one', async () => {
  const eli = await signUp('eli');
  const first = (await upload(eli, phoneJpeg())).data.id;
  const second = (await upload(eli, phoneJpeg())).data.id;
  await setPicture(eli, 'cover', first);
  await setPicture(eli, 'cover', second);
  assert.equal((await fetchImage(anon, first)).status, 404);
  assert.equal((await eli.call('GET', '/me')).data.user.cover, second);
  await setPicture(eli, 'cover', null);
  assert.equal((await fetchImage(anon, second)).status, 404);
  assert.equal((await eli.call('GET', '/me')).data.user.cover, null);
});

test('a picture cannot be stolen, reused, or moved into a post', async () => {
  const gal = await signUp('gal');
  const own = (await upload(gal, phoneJpeg())).data.id;
  assert.equal((await setPicture(alice, 'avatar', own)).status, 400);
  assert.equal((await setPicture(gal, 'nonsense', own)).status, 400);
  await setPicture(gal, 'avatar', own);
  assert.equal((await setPicture(gal, 'cover', own)).status, 400);
  assert.equal((await gal.call('POST', '/posts', { body: '', audience: 'public', images: [own] })).status, 400);
});

test('a profile shows a stranger only public posts, and a friend everything', async () => {
  const hila = await signUp('hila');
  const ido = await signUp('ido');
  await hila.call('POST', '/posts', { body: 'לכולם', audience: 'public' });
  await hila.call('POST', '/posts', { body: 'לחברים', audience: 'friends' });
  const stranger = (await ido.call('GET', `/users/${hila.id}`)).data;
  assert.deepEqual(stranger.posts.map((post) => post.body), ['לכולם']);
  assert.equal(stranger.friendship, null);
  await ido.call('POST', `/friends/${hila.id}`);
  assert.equal((await ido.call('GET', `/users/${hila.id}`)).data.friendship, 'outgoing');
  assert.equal((await hila.call('GET', `/users/${ido.id}`)).data.friendship, 'incoming');
  await hila.call('POST', `/friends/${ido.id}`);
  const friend = (await ido.call('GET', `/users/${hila.id}`)).data;
  assert.deepEqual(friend.posts.map((post) => post.body), ['לחברים', 'לכולם']);
  assert.equal(friend.friendship, 'friends');
  assert.equal((await hila.call('GET', `/users/${hila.id}`)).data.friendship, 'self');
  assert.equal((await client()('GET', `/users/${hila.id}`)).status, 401);
  assert.equal((await ido.call('GET', '/users/99999')).status, 404);
});

test('uploads stop at the storage limit, for everyone', async () => {
  const first = await signUp('first', STORAGE_PORT);
  const second = await signUp('second', STORAGE_PORT);
  assert.equal((await upload(first, phoneJpeg())).status, 201);
  const refused = await upload(second, phoneJpeg());
  assert.equal(refused.status, 429);
  assert.match(refused.data.error, /מגבלת האחסון/);
});

test('uploads stop at the daily limit of the whole network', async () => {
  const first = await signUp('first', UPLOADS_PORT);
  const second = await signUp('second', UPLOADS_PORT);
  assert.equal((await upload(first, phoneJpeg())).status, 201);
  assert.equal((await upload(second, phoneJpeg())).status, 201);
  assert.equal((await upload(first, phoneJpeg())).status, 429);
});

// One admin for all invite-only tests: logins are rate limited per address, as in production.
let bossAccount;
test('a place reaches friends only, rounded to about a kilometre', async () => {
  const dana = await signUp('dana-globe');
  const eli = await signUp('eli-globe');
  const stranger = await signUp('stranger-globe');
  await dana.call('POST', `/friends/${eli.id}`);
  await eli.call('POST', `/friends/${dana.id}`);

  const mine = () => eli.call('GET', '/globe').then(({ data }) => data.friends.find((friend) => friend.id === dana.id));
  assert.equal((await mine()).place, null);
  assert.equal((await dana.call('GET', '/me')).data.user.place, null);

  const saved = await dana.call('PUT', '/me/place', { lat: 32.471234, lng: 34.976789 });
  assert.deepEqual(saved.data.place, { lat: 32.47, lng: 34.98 });
  assert.deepEqual((await mine()).place, { lat: 32.47, lng: 34.98 });
  assert.deepEqual((await dana.call('GET', '/me')).data.user.place, { lat: 32.47, lng: 34.98 });

  const others = (await stranger.call('GET', '/globe')).data.friends;
  assert.equal(others.length, 0);
  assert.equal(JSON.stringify((await stranger.call('GET', `/users/${dana.id}`)).data).includes('32.47'), false);
  assert.equal((await anon('GET', '/globe')).status, 401);
});

test('a what\'s up line reaches friends only, and deleting it erases it', async () => {
  const dana = await signUp('dana-status');
  const eli = await signUp('eli-status');
  const stranger = await signUp('stranger-status');
  await dana.call('POST', `/friends/${eli.id}`);
  await eli.call('POST', `/friends/${dana.id}`);

  assert.equal((await dana.call('PUT', '/me/status', { text: 'x'.repeat(141) })).status, 400);
  assert.equal((await dana.call('PUT', '/me/status', { text: '  ' })).status, 400);
  assert.equal((await anon('PUT', '/me/status', { text: 'שלום' })).status, 401);
  assert.equal((await dana.call('PUT', '/me/status', { text: ' עברנו דירה ' })).data.status.text, 'עברנו דירה');
  assert.equal((await dana.call('GET', '/me')).data.user.status.text, 'עברנו דירה');

  const seenBy = async (who) => (await who.call('GET', '/update')).data.friends.statuses.map((item) => item.text);
  assert.deepEqual(await seenBy(eli), ['עברנו דירה']);
  assert.deepEqual(await seenBy(stranger), []);
  assert.deepEqual(await seenBy(dana), []);
  assert.equal((await eli.call('GET', '/globe')).data.friends[0].status, 'עברנו דירה');

  assert.equal((await dana.call('DELETE', '/me/status')).status, 200);
  assert.deepEqual(await seenBy(eli), []);
  assert.equal((await dana.call('GET', '/me')).data.user.status, null);
  assert.equal((await eli.call('GET', '/globe')).data.friends[0].status, null);
});

test('the update tells what friends did: new posts, replies, a new place, a new member', async () => {
  const dana = await signUp('dana-update');
  const eli = await signUp('eli-update');
  const stranger = await signUp('stranger-update');
  await dana.call('POST', `/friends/${eli.id}`);
  await eli.call('POST', `/friends/${dana.id}`);
  assert.equal((await anon('GET', '/update')).status, 401);

  const mine = (await eli.call('POST', '/posts', { body: 'הפוסט שלי', audience: 'public' })).data.id;
  const first = (await dana.call('POST', '/posts', { body: 'ראשון', audience: 'friends' })).data.id;
  await dana.call('POST', '/posts', { body: 'שני', audience: 'friends' });
  await dana.call('POST', `/posts/${mine}/comments`, { body: 'תגובה' });
  await stranger.call('POST', `/posts/${mine}/comments`, { body: 'גם אני' });
  await eli.call('POST', `/posts/${mine}/comments`, { body: 'אני עונה לעצמי' });
  await dana.call('PUT', '/me/place', { lat: 48.85, lng: 2.35 });

  const { friends } = (await eli.call('GET', `/update?seen=${first}`)).data;
  assert.deepEqual(friends.fresh, [{ user: { id: dana.id, name: 'dana-update', avatar: null }, count: 1 }]);
  assert.deepEqual(friends.replies.map((item) => [item.user.id, item.postId, item.onMine]), [[stranger.id, mine, true], [dana.id, mine, true]]);
  assert.deepEqual(friends.moved.map((item) => [item.user.id, item.place]), [[dana.id, { lat: 48.85, lng: 2.35 }]]);
  assert.deepEqual(friends.joined.map((item) => item.user.id), [dana.id]);
  assert.deepEqual(friends.quiet, []);
  assert.equal((await eli.call('GET', '/update')).data.friends.fresh[0].count, 2);

  // What others felt about my post and about my comment, not what I felt myself, newest first.
  const answer = (await eli.call('GET', `/posts/${mine}`)).data.comments.at(-1).id;
  await dana.call('PUT', `/posts/${mine}/reaction`, { kind: 'love' });
  await stranger.call('PUT', `/comments/${answer}/reaction`, { kind: '🌻' });
  await eli.call('PUT', `/posts/${mine}/reaction`, { kind: 'funny' });
  await eli.call('PUT', `/posts/${first}/reaction`, { kind: 'funny' });
  const felt = (await eli.call('GET', '/update')).data.friends.felt;
  assert.deepEqual(felt.map((item) => [item.user.id, item.kind, item.postId, item.onPost]), [[stranger.id, '🌻', mine, false], [dana.id, 'love', mine, true]]);
  assert.deepEqual((await dana.call('GET', '/update')).data.friends.felt.map((item) => [item.user.id, item.kind, item.postId]), [[eli.id, 'funny', first]]);

  // Nothing about people who are not friends, and nothing once the place is switched off.
  const outside = (await stranger.call('GET', '/update')).data.friends;
  assert.deepEqual([outside.fresh, outside.moved, outside.joined, outside.replies], [[], [], [], []]);
  await dana.call('DELETE', '/me/place');
  assert.deepEqual((await eli.call('GET', '/update')).data.friends.moved, []);
});

test('news from outside is read by the server, cleaned to plain text, and kept', async () => {
  const eli = await signUp('eli-news');
  const { community, country, world } = (await eli.call('GET', '/update')).data;
  assert.deepEqual(community.events.map((event) => [event.name, event.url, event.time, event.venue]),
    [['ערב ג\'אז', 'https://events.example/1', '20:30', 'בית הראשונים'], ['מאוחר יותר באותו יום', null, '20:30', 'בית הראשונים']]);
  // Pictures come through this server, and only from the hosts it trusts.
  const picture = (item) => item.image && new URL(item.image, 'http://x').searchParams.get('src');
  assert.equal(picture(community.events[0]), `https://pardespedia.info/index.php?title=Special:Redirect/file/${encodeURIComponent("ערב ג'אז.jpg")}&width=480`);
  assert.equal(community.events[1].image, null);
  assert.deepEqual(community.plans.map((plan) => [plan.title, plan.adds, plan.url.split('#')[1]]), [['דרך הבנים', '12 יחידות דיור', 'p-308-1']]);
  assert.deepEqual(country, [
    { text: 'גשם ירד & ירד בצפון', url: 'https://he.wikipedia.org/wiki/%D7%92%D7%A9%D7%9D', image: `/api/picture?src=${encodeURIComponent('https://upload.wikimedia.org/rain.jpg')}` },
    { text: 'ידיעה שנייה', url: null, image: null }]);
  assert.deepEqual(world, [{ text: 'The sea sees a big wave.', url: 'https://en.wikipedia.org/wiki/Big_Wave', image: null }]);

  // Israel: one headline from each channel in turn, newest first within each, a failing channel skipped.
  const { israel } = (await eli.call('GET', '/update')).data;
  assert.deepEqual(israel.map((item) => [item.channel, item.text, item.url, picture(item)]), [
    ['N12', 'ידיעה חדשה ב-N12 & עוד', 'https://n12.example/5', 'https://img.mako.co.il/a.jpg'],
    ['עכשיו 14', 'כותרת בערוץ 14', 'https://c14.example/10', 'https://www.c14.co.il/b.jpg'],
    ['כאן חדשות', 'כותרת בכאן', 'https://kan.example/20', null],
    ['N12', 'ידיעה ישנה ב-N12', 'https://n12.example/90', null]]);

  // The picture route passes on nothing from elsewhere, and nothing to a stranger.
  assert.equal((await eli.call('GET', `/picture?src=${encodeURIComponent('https://evil.example/x.jpg')}`)).status, 404);
  assert.equal((await eli.call('GET', `/picture?src=${encodeURIComponent('http://upload.wikimedia.org/x.jpg')}`)).status, 404);
  assert.equal((await anon('GET', `/picture?src=${encodeURIComponent('https://upload.wikimedia.org/x.jpg')}`)).status, 401);

  const fetched = digestRequests.length;
  await eli.call('GET', '/update');
  assert.equal(digestRequests.length, fetched);
});

test('switching sharing off erases the place, and ending a friendship ends the view', async () => {
  const dana = await signUp('dana-globe2');
  const eli = await signUp('eli-globe2');
  await dana.call('POST', `/friends/${eli.id}`);
  await eli.call('POST', `/friends/${dana.id}`);
  await dana.call('PUT', '/me/place', { lat: 10, lng: 20 });
  assert.equal((await dana.call('DELETE', '/me/place')).status, 200);
  assert.equal((await eli.call('GET', '/globe')).data.friends[0].place, null);
  assert.equal((await dana.call('GET', '/me')).data.user.place, null);

  await dana.call('PUT', '/me/place', { lat: 10, lng: 20 });
  await eli.call('DELETE', `/friends/${dana.id}`);
  assert.equal((await eli.call('GET', '/globe')).data.friends.length, 0);
});

test('a place must be a real point on the earth', async () => {
  const { call } = await signUp('fay-globe');
  for (const body of [{}, { lat: '32', lng: 34 }, { lat: 91, lng: 0 }, { lat: 0, lng: 181 }, { lat: null, lng: null }]) {
    assert.equal((await call('PUT', '/me/place', body)).status, 400, JSON.stringify(body));
  }
  assert.equal((await anon('PUT', '/me/place', { lat: 1, lng: 1 })).status, 401);
});

// Two friends and someone who is a friend of neither, for the private messages.
async function threeForMessages(tag) {
  const dana = await signUp(`dana-${tag}`);
  const eli = await signUp(`eli-${tag}`);
  const stranger = await signUp(`stranger-${tag}`);
  await dana.call('POST', `/friends/${eli.id}`);
  await eli.call('POST', `/friends/${dana.id}`);
  return { dana, eli, stranger };
}
const write = (from, to, body, image) => from.call('POST', `/messages/${to.id}`, { body, image });
const talk = (reader, other, query = '') => reader.call('GET', `/messages/${other.id}${query}`).then(({ data }) => data);
const unread = (user) => user.call('GET', '/me').then(({ data }) => data.user.unreadMessages);

test('a message reaches the friend it was written to, and nobody else', async () => {
  const { dana, eli, stranger } = await threeForMessages('msg');
  assert.equal((await anon('GET', '/messages')).status, 401);
  assert.equal((await anon('GET', `/messages/${eli.id}`)).status, 401);
  assert.equal((await anon('POST', `/messages/${eli.id}`, { body: 'x' })).status, 401);
  assert.deepEqual((await talk(dana, eli)).messages, []);
  assert.deepEqual((await dana.call('GET', '/messages')).data.conversations, []);

  const sent = await write(dana, eli, ' שלום אלי ');
  assert.equal(sent.status, 201);
  assert.equal(sent.data.message.body, 'שלום אלי');
  await write(eli, dana, 'שלום דנה https://example.com/a');

  const mine = await talk(dana, eli);
  assert.equal(mine.user.id, eli.id);
  assert.equal(mine.canSend, true);
  assert.deepEqual(mine.messages.map((item) => [item.body, item.mine]), [['שלום אלי', true], ['שלום דנה https://example.com/a', false]]);
  assert.deepEqual((await talk(eli, dana)).messages.map((item) => item.mine), [false, true]);

  const list = (await eli.call('GET', '/messages')).data.conversations;
  assert.deepEqual(list.map((item) => item.user.id), [dana.id]);
  assert.deepEqual([list[0].last.body, list[0].last.mine], ['שלום דנה https://example.com/a', true]);

  // A stranger gets an empty conversation of their own, never this one.
  assert.deepEqual((await stranger.call('GET', '/messages')).data.conversations, []);
  const outside = await talk(stranger, dana);
  assert.deepEqual([outside.messages, outside.canSend], [[], false]);
  assert.equal((await write(stranger, dana, 'היי')).status, 403);
  assert.equal((await stranger.call('PUT', `/messages/${dana.id}/read`, {})).status, 404);
  assert.equal((await stranger.call('DELETE', `/messages/${dana.id}/${sent.data.message.id}`)).status, 404);
  assert.equal(JSON.stringify((await stranger.call('GET', '/me')).data).includes('שלום'), false);

  assert.equal((await write(dana, dana, 'לעצמי')).status, 404);
  assert.equal((await write(dana, { id: 999999 }, 'לאף אחד')).status, 404);
  assert.equal((await write(dana, eli, '   ')).status, 400);
  assert.equal((await write(dana, eli, 'x'.repeat(50001))).status, 400);
  assert.equal((await talk(dana, eli)).messages.length, 2);
});

test('unread messages are counted until they are read, and the mark never runs ahead', async () => {
  const { dana, eli } = await threeForMessages('read');
  const first = (await write(dana, eli, 'אחת')).data.message.id;
  await write(dana, eli, 'שתיים');
  assert.equal(await unread(eli), 2);
  assert.equal(await unread(dana), 0);
  assert.equal((await eli.call('GET', '/messages')).data.conversations[0].unread, 2);
  assert.equal((await dana.call('GET', '/messages')).data.conversations[0].unread, 0);

  // Opening the conversation does not mark it by itself. The screen says how far it showed.
  assert.equal((await talk(eli, dana)).readUpTo, 0);
  assert.equal(await unread(eli), 2);
  assert.equal((await eli.call('PUT', `/messages/${dana.id}/read`, { upTo: first })).status, 200);
  assert.equal(await unread(eli), 1);
  await eli.call('PUT', `/messages/${dana.id}/read`, { upTo: 10 ** 12 });
  assert.equal(await unread(eli), 0);
  await write(dana, eli, 'שלוש');
  assert.equal(await unread(eli), 1);

  // Whoever answers has seen what they answer.
  await write(eli, dana, 'עונה');
  assert.equal(await unread(eli), 0);
  assert.equal(await unread(dana), 1);
  assert.equal((await eli.call('PUT', '/messages/999999/read', {})).status, 404);
});

test('a conversation comes in pages, and asking again brings only what is new', async () => {
  const { dana, eli } = await threeForMessages('pages');
  for (let i = 0; i < 53; i++) await write(dana, eli, `הודעה ${i}`);
  const page = await talk(eli, dana);
  assert.equal(page.messages.length, 50);
  assert.equal(page.hasMore, true);
  assert.equal(page.messages.at(-1).body, 'הודעה 52');
  const older = await talk(eli, dana, `?before=${page.messages[0].id}`);
  assert.deepEqual(older.messages.map((item) => item.body), ['הודעה 0', 'הודעה 1', 'הודעה 2']);
  assert.equal(older.hasMore, false);

  const newest = page.messages.at(-1).id;
  assert.deepEqual((await talk(eli, dana, `?after=${newest}&since=${page.now}`)).messages, []);
  await write(eli, dana, 'חדשה');
  const fresh = await talk(dana, eli, `?after=${newest}&since=${page.now}`);
  assert.deepEqual(fresh.messages.map((item) => [item.body, item.mine]), [['חדשה', false]]);
});

test('a picture in a message opens for the two of them only, and cannot be used again', async () => {
  const { dana, eli, stranger } = await threeForMessages('picture');
  const image = (await upload(dana, phoneJpeg())).data.id;
  const sent = (await write(dana, eli, '', image)).data.message;
  assert.deepEqual([sent.body, sent.image], [null, image]);
  assert.equal((await talk(eli, dana)).messages[0].image, image);
  assert.equal((await eli.call('GET', '/messages')).data.conversations[0].last.image, true);

  assert.equal((await fetchImage(eli.call, image)).status, 200);
  assert.equal((await fetchImage(dana.call, image)).status, 200);
  assert.equal((await fetchImage(stranger.call, image)).status, 404);
  assert.equal((await fetchImage(anon, image)).status, 404);

  assert.equal((await write(dana, eli, 'שוב', image)).status, 400);
  assert.equal((await dana.call('POST', '/posts', { body: 'x', audience: 'public', images: [image] })).status, 400);
  assert.equal((await dana.call('PUT', '/me/images', { kind: 'avatar', imageId: image })).status, 400);
  const theirs = (await upload(eli, phoneJpeg())).data.id;
  assert.equal((await write(dana, eli, 'לא שלי', theirs)).status, 400);
});

test('a removed message leaves a mark for both sides, and its picture is really deleted', async () => {
  const { dana, eli } = await threeForMessages('gone');
  const image = (await upload(dana, phoneJpeg())).data.id;
  const sent = (await write(dana, eli, 'תראה', image)).data.message;
  const before = await talk(eli, dana);
  assert.equal(await unread(eli), 1);

  assert.equal((await eli.call('DELETE', `/messages/${dana.id}/${sent.id}`)).status, 404);
  assert.equal((await dana.call('DELETE', `/messages/${eli.id}/${sent.id}`)).status, 200);
  assert.equal((await dana.call('DELETE', `/messages/${eli.id}/${sent.id}`)).status, 404);
  for (const [reader, other] of [[eli, dana], [dana, eli]]) {
    const [after] = (await talk(reader, other)).messages;
    assert.deepEqual([after.id, after.body, after.image, after.removed], [sent.id, null, null, true]);
  }
  assert.equal((await fetchImage(dana.call, image)).status, 404);
  assert.equal((await fetchImage(eli.call, image)).status, 404);
  assert.equal(await unread(eli), 0);
  assert.deepEqual((await eli.call('GET', '/messages')).data.conversations[0].last, { mine: false, body: null, image: false, at: before.messages[0].at, removed: true });

  // An open conversation learns of the removal the next time it asks what is new.
  assert.deepEqual((await talk(eli, dana, `?after=${sent.id}&since=${before.now}`)).removed, [sent.id]);
  assert.deepEqual((await talk(eli, dana, `?after=${sent.id}&since=${Date.now() + 60000}`)).removed, []);
});

test('ending a friendship stops the writing, and what was written stays readable', async () => {
  const { dana, eli } = await threeForMessages('apart');
  await write(dana, eli, 'לפני');
  await eli.call('DELETE', `/friends/${dana.id}`);
  assert.equal((await write(dana, eli, 'אחרי')).status, 403);
  assert.equal((await write(eli, dana, 'אחרי')).status, 403);
  const kept = await talk(eli, dana);
  assert.deepEqual([kept.canSend, kept.messages.map((item) => item.body)], [false, ['לפני']]);
  assert.equal((await eli.call('PUT', `/messages/${dana.id}/read`, {})).status, 200);
  assert.equal(await unread(eli), 0);
});

async function theBoss() {
  bossAccount ??= await signUp('boss', INVITE_PORT);
  return bossAccount;
}

async function openInvite(link) {
  const call = client(INVITE_PORT);
  const verified = await call('POST', '/auth/verify', { token: link.split('#token=')[1] });
  return { call, verified };
}

test('the admin invites a member, and the link works once', async () => {
  const boss = await theBoss();
  assert.equal((await boss.call('GET', '/me')).data.user.admin, true);
  const { status, data } = await boss.call('POST', '/admin/invites', {});
  assert.equal(status, 201);
  const member = await openInvite(data.link);
  assert.equal(member.verified.status, 200);
  await member.call('PUT', '/me', { name: 'חברה' });
  assert.equal((await member.call('GET', '/me')).data.user.admin, false);
  assert.equal((await openInvite(data.link)).verified.status, 400);
});

test('someone who joins by invitation starts as a friend of the admin', async () => {
  const boss = await theBoss();
  const { data } = await boss.call('POST', '/admin/invites', {});
  const { call } = await openInvite(data.link);
  assert.equal((await boss.call('GET', '/friends')).data.friends.some((user) => user.name === ''), false);
  await call('PUT', '/me', { name: 'מוזמנת' });
  await boss.call('POST', '/posts', { body: 'ברוכה הבאה', audience: 'friends' });
  assert.equal((await call('GET', '/feed')).data.posts[0].body, 'ברוכה הבאה');
  assert.ok((await boss.call('GET', '/friends')).data.friends.some((user) => user.name === 'מוזמנת'));
});

test('the waiting friend requests are counted, for the addressee only', async () => {
  const shir = await signUp('shir');
  const ziv = await signUp('ziv');
  await shir.call('POST', `/friends/${ziv.id}`);
  assert.equal((await ziv.call('GET', '/me')).data.user.friendRequests, 1);
  assert.equal((await shir.call('GET', '/me')).data.user.friendRequests, 0);
  await ziv.call('POST', `/friends/${shir.id}`);
  assert.equal((await ziv.call('GET', '/me')).data.user.friendRequests, 0);
});

// Plays both redirects of a Google sign-in, with `code` standing for who chose to sign in.
async function googleFlow(code, { user, link = false, mail = false, quiet = false, forgedState = false } = {}) {
  const base = `http://localhost:${INVITE_PORT}/api/auth/google`;
  const session = user?.call.cookie() ?? '';
  const start = await fetch(`${base}${link ? '?link=1' : mail ? '?mail=1' : quiet ? '?quiet=1' : ''}`, { redirect: 'manual', headers: { Cookie: session } });
  const asked = new URL(start.headers.get('location'), base);
  const state = asked.searchParams.get('state');
  if (!state) return { result: null, asked };
  const stateCookie = start.headers.get('set-cookie')?.split(';')[0] ?? '';
  // Google answers a quiet request it cannot settle without the person with an error, not a code.
  const answer = code ? `code=${code}` : 'error=interaction_required';
  const back = await fetch(`${base}/callback?${answer}&state=${forgedState ? 'forged' : state}`, {
    redirect: 'manual',
    headers: { Cookie: [session, stateCookie].filter(Boolean).join('; ') },
  });
  const cookie = back.headers.getSetCookie().find((line) => line.startsWith('session='))?.split(';')[0];
  return { result: new URL(back.headers.get('location'), base).searchParams.get('google'), cookie, asked };
}

test('a linked Google account opens the same account on another device', async () => {
  const lior = await signUp('lior', INVITE_PORT);
  assert.equal((await googleFlow('google-lior', { user: lior, link: true })).result, 'linked');
  assert.equal((await lior.call('GET', '/me')).data.user.googleLinked, true);
  const { result, cookie } = await googleFlow('google-lior');
  assert.equal(result, 'login');
  const me = await fetch(`http://localhost:${INVITE_PORT}/api/me`, { headers: { Cookie: cookie } });
  assert.equal((await me.json()).user.id, lior.id);
});

test('Google never creates an account, and a forged redirect gets nothing', async () => {
  const boss = await theBoss();
  const before = (await boss.call('GET', '/admin/overview')).data.users;
  const unknown = await googleFlow('google-stranger');
  assert.equal(unknown.result, 'unknown');
  assert.equal(unknown.cookie, undefined);
  assert.equal((await boss.call('GET', '/admin/overview')).data.users, before);
  const forged = await googleFlow('google-lior', { forgedState: true });
  assert.equal(forged.result, 'failed');
  assert.equal(forged.cookie, undefined);
  assert.equal((await googleFlow('wrong-audience')).result, 'failed');
});

test('the installed app lets in quietly a member Google vouches for, and says nothing otherwise', async () => {
  const boss = await theBoss();
  const journalOf = (...more) => execFileSync('node', ['tools/logs.mjs', '--persist-to', join(root, String(INVITE_PORT)), '--journal', ...more], { encoding: 'utf8' });
  const quiet = await googleFlow('google-lior', { quiet: true });
  assert.equal(quiet.asked.searchParams.get('prompt'), 'none');
  assert.equal(quiet.result, 'login');
  assert.ok(quiet.cookie);
  for (const code of ['google-stranger', null, 'wrong-audience']) {
    const failed = await googleFlow(code, { quiet: true });
    assert.equal(failed.result, 'quiet');
    assert.equal(failed.cookie, undefined);
  }
  // A waiting invitation is taken only by someone who chose to sign in.
  await boss.call('POST', '/admin/email-invites', { email: 'quiet.one@gmail.com', name: 'שקטה' });
  assert.equal((await googleFlow('google-quiet:quietone@gmail.com', { quiet: true })).result, 'quiet');
  assert.equal((await googleFlow('google-quiet:quietone@gmail.com')).result, 'joined');
  // What Google answered each time is in the journal, with the reason when it gave one.
  const told = journalOf('--kind', 'google', '--limit', '6');
  assert.match(told, /quiet: login\n/);
  assert.match(told, /quiet: failed \(interaction_required\)/);
  assert.match(told, /quiet: unknown/);
  assert.match(told, /login: joined/);
  assert.match(journalOf('--kind', 'login', '--limit', '8'), /google, quiet/);
});

test('one Google account belongs to one person, and it can be unlinked', async () => {
  const maya = await signUp('maya', INVITE_PORT);
  assert.equal((await googleFlow('google-lior', { user: maya, link: true })).result, 'taken');
  assert.equal((await googleFlow('google-maya', { user: maya, link: true })).result, 'linked');
  assert.equal((await maya.call('DELETE', '/me/google')).status, 200);
  assert.equal((await googleFlow('google-maya')).result, 'unknown');
});

test('an email invitation carries the name, works once, and makes a friend of the admin', async () => {
  const boss = await theBoss();
  const { status, data } = await boss.call('POST', '/admin/email-invites', { email: 'Dana.Cohen@Gmail.com', name: 'דנה כהן' });
  assert.equal(status, 201);
  assert.equal(data.email, 'danacohen@gmail.com');
  assert.deepEqual((await boss.call('GET', '/admin/overview')).data.emailInvites.map((item) => [item.email, item.name]), [['danacohen@gmail.com', 'דנה כהן']]);
  const dana = await openInvite(data.link);
  assert.equal(dana.verified.data.joined, true);
  const me = (await dana.call('GET', '/me')).data.user;
  assert.equal(me.name, 'דנה כהן');
  assert.deepEqual((await dana.call('GET', '/friends')).data.friends.map((user) => user.name), ['boss']);
  assert.equal((await boss.call('GET', '/admin/overview')).data.emailInvites.length, 0);
  assert.equal((await openInvite(data.link)).verified.status, 400);
  assert.equal((await boss.call('POST', '/admin/email-invites', { email: 'dana.cohen@gmail.com' })).status, 409);
});

test('inviting an address again replaces the link that was waiting', async () => {
  const boss = await theBoss();
  const first = (await boss.call('POST', '/admin/email-invites', { email: 'twice@example.org', name: 'פעמיים' })).data;
  const second = (await boss.call('POST', '/admin/email-invites', { email: 'twice@example.org' })).data;
  assert.equal((await boss.call('GET', '/admin/overview')).data.emailInvites.length, 1);
  assert.equal((await openInvite(first.link)).verified.status, 400);
  const member = await openInvite(second.link);
  assert.equal(member.verified.status, 200);
  assert.equal((await member.call('GET', '/me')).data.user.name, '');
});

test('someone invited by email joins with Google at that address, with the picture of the account', async () => {
  const boss = await theBoss();
  await boss.call('POST', '/admin/email-invites', { email: 'New.Friend@Gmail.com', name: 'חבר חדש' });
  const joined = await googleFlow('google-new:newfriend@gmail.com:photo');
  assert.equal(joined.result, 'joined');
  const headers = { Cookie: joined.cookie };
  const me = await (await fetch(`http://localhost:${INVITE_PORT}/api/me`, { headers })).json();
  assert.equal(me.user.googleLinked, true);
  assert.equal(me.user.name, 'חבר חדש');
  const picture = await fetch(`http://localhost:${INVITE_PORT}/api/images/${me.user.avatar}`);
  assert.equal(picture.status, 200);
  const stored = Buffer.from(await picture.arrayBuffer());
  assert.ok(!stored.includes('GPS'));
  assert.deepEqual(pictureRequests, ['/a-/photo=s400-c']);
  const friends = await (await fetch(`http://localhost:${INVITE_PORT}/api/friends`, { headers })).json();
  assert.deepEqual(friends.friends.map((user) => user.name), ['boss']);
  assert.equal((await boss.call('GET', '/admin/overview')).data.emailInvites.length, 0);
  assert.equal((await googleFlow('google-new:newfriend@gmail.com')).result, 'login');
  assert.equal((await googleFlow('google-other:newfriend@gmail.com')).result, 'unknown');
});

test('a member who came in by the emailed link returns with Google at the same address', async () => {
  const boss = await theBoss();
  const { link } = (await boss.call('POST', '/admin/email-invites', { email: 'back@gmail.com', name: 'חוזרת' })).data;
  const member = await openInvite(link);
  const { id } = (await member.call('GET', '/me')).data.user;
  const back = await googleFlow('google-back:back@gmail.com:letter');
  assert.equal(back.result, 'login');
  const me = await (await fetch(`http://localhost:${INVITE_PORT}/api/me`, { headers: { Cookie: back.cookie } })).json();
  assert.equal(me.user.id, id);
  assert.equal(me.user.googleLinked, true);
  // Google's drawn letter is not a photo, so the member keeps the picture made of their name.
  assert.equal(me.user.avatar, null);
});

test('linking Google brings its picture only to someone who has none', async () => {
  const noPicture = await signUp('pictureless', INVITE_PORT);
  assert.equal((await googleFlow('google-pictureless:p@gmail.com:photo', { user: noPicture, link: true })).result, 'linked');
  assert.ok((await noPicture.call('GET', '/me')).data.user.avatar);
  const chosen = await signUp('chosen', INVITE_PORT);
  const own = (await upload(chosen, phoneJpeg())).data.id;
  await chosen.call('PUT', '/me/images', { kind: 'avatar', imageId: own });
  await googleFlow('google-chosen:c@gmail.com:photo', { user: chosen, link: true });
  assert.equal((await chosen.call('GET', '/me')).data.user.avatar, own);
  const elsewhere = await signUp('elsewhere', INVITE_PORT);
  await googleFlow('google-elsewhere:e@gmail.com:elsewhere', { user: elsewhere, link: true });
  assert.equal((await elsewhere.call('GET', '/me')).data.user.avatar, null);
});

test('an email invitation needs a verified address, and only the admin can send or cancel one', async () => {
  const boss = await theBoss();
  await boss.call('POST', '/admin/email-invites', { email: 'careful@gmail.com' });
  assert.equal((await googleFlow('google-careful:careful@gmail.com!')).result, 'unknown');
  const stranger = await signUp('stranger', INVITE_PORT);
  assert.equal((await stranger.call('POST', '/admin/email-invites', { email: 'x@gmail.com' })).status, 403);
  assert.equal((await stranger.call('DELETE', '/admin/email-invites/careful%40gmail.com')).status, 403);
  assert.equal((await boss.call('POST', '/admin/email-invites', { email: 'not an address' })).status, 400);
  assert.equal((await boss.call('POST', '/admin/email-invites', { email: 'x@invite.invalid' })).status, 400);
  await boss.call('DELETE', '/admin/email-invites/careful%40gmail.com');
  assert.equal((await googleFlow('google-careful:careful@gmail.com')).result, 'unknown');
});

// The parts of a letter as the recipient's mail program would show them.
function readLetter(message) {
  const [head, ...parts] = message.split(/\r\n--part-[\w-]+(?:--)?(?:\r\n)?/);
  const decode = (part) => Buffer.from(part.split('\r\n\r\n')[1].replace(/\r\n/g, ''), 'base64').toString();
  const subject = Buffer.from(head.match(/Subject: =\?UTF-8\?B\?(.+)\?=/)[1], 'base64').toString();
  return { to: head.match(/^To: (.+)$/m)[1], subject, text: decode(parts[0]), html: decode(parts[1]) };
}

test('until the admin connects a Gmail, an invitation comes back as a letter to send by hand', async () => {
  const boss = await theBoss();
  const { data } = await boss.call('POST', '/admin/email-invites', { email: 'by.hand@example.org', name: 'ביד רמה' });
  assert.equal(data.sent, false);
  assert.equal(data.sendProblem, null);
  assert.equal(data.subject, 'הזמנה אישית לרשת החופשית');
  assert.ok(data.text.startsWith('היי ביד,'));
  assert.ok(data.text.includes(data.link) && data.text.includes('/intro'));
  assert.equal((await boss.call('GET', '/admin/overview')).data.mailer, null);
});

test('only the admin connects a Gmail, and only for sending', async () => {
  const boss = await theBoss();
  const stranger = await signUp('nosy', INVITE_PORT);
  assert.equal((await googleFlow('google-nosy:nosy@gmail.com', { user: stranger, mail: true })).result, null);
  const unticked = await googleFlow('google-boss-unticked:boss@gmail.com', { user: boss, mail: true });
  assert.equal(unticked.result, 'mail-refused');
  assert.equal((await boss.call('GET', '/admin/overview')).data.mailer, null);
  const connected = await googleFlow('google-boss:The.Boss@gmail.com', { user: boss, mail: true });
  assert.equal(connected.result, 'mail');
  assert.equal(connected.asked.searchParams.get('scope'), 'openid email https://www.googleapis.com/auth/gmail.send '
    + 'https://www.googleapis.com/auth/contacts.readonly https://www.googleapis.com/auth/contacts.other.readonly');
  assert.equal(connected.asked.searchParams.get('access_type'), 'offline');
  // The boss ticked only the sending, so nothing is read from the contacts and Google is not asked.
  assert.deepEqual((await boss.call('GET', '/admin/overview')).data.mailer, { address: 'The.Boss@gmail.com', contacts: false });
  assert.deepEqual((await boss.call('GET', '/admin/contacts')).data, { contacts: null, problem: 'unconnected' });
  assert.equal(peopleRequests.length, 0);
  // Connecting for sending is not signing in: no account was linked or created by it.
  assert.equal((await boss.call('GET', '/me')).data.user.googleLinked, false);
});

test('with a Gmail connected the invitation is sent, right to left, to the address as typed', async () => {
  const boss = await theBoss();
  const { data } = await boss.call('POST', '/admin/email-invites', { email: 'Sarig.Test@gmail.com', name: 'שריג <b>', note: 'שורה ראשונה\n\nשורה & שנייה' });
  assert.equal(data.sent, true);
  const sent = mailSent.at(-1);
  assert.equal(sent.pass, 'Bearer pass-for-key-of-google-boss');
  const letter = readLetter(sent.message);
  assert.equal(letter.to, 'Sarig.Test@gmail.com');
  assert.equal(letter.subject, 'הזמנה אישית לרשת החופשית');
  assert.equal(letter.text, data.text);
  assert.ok(letter.text.includes('שורה & שנייה') && letter.text.includes(data.link));
  assert.match(letter.html, /^<div dir="rtl"[^>]*text-align:right/);
  assert.ok(letter.html.includes('שורה &amp; שנייה') && letter.html.includes('היי שריג,'));
  assert.ok(letter.html.includes(`href="${data.link}"`) && letter.html.includes('/video/intro.jpg'));
  assert.ok(!letter.html.includes('<b>'));
  // Asking not to send leaves the letter with the admin.
  const before = mailSent.length;
  assert.equal((await boss.call('POST', '/admin/email-invites', { email: 'quiet@example.org', send: false })).data.sent, false);
  assert.equal(mailSent.length, before);
  assert.equal((await boss.call('POST', '/admin/email-invites', { email: 'long@example.org', note: 'א'.repeat(1501) })).status, 400);
});

test('disconnecting erases the key and tells Google, and a withdrawn key is forgotten', async () => {
  const boss = await theBoss();
  assert.equal((await boss.call('DELETE', '/admin/mailer')).status, 200);
  assert.deepEqual(revoked, ['key-of-google-boss']);
  assert.equal((await boss.call('GET', '/admin/overview')).data.mailer, null);
  assert.equal((await googleFlow('google-boss-withdrawn:boss@gmail.com', { user: boss, mail: true })).result, 'mail');
  const before = mailSent.length;
  const { data } = await boss.call('POST', '/admin/email-invites', { email: 'late@example.org' });
  assert.equal(data.sent, false);
  assert.equal(data.sendProblem, 'expired');
  assert.ok(data.text.includes(data.link));
  assert.equal(mailSent.length, before);
  assert.equal((await boss.call('GET', '/admin/overview')).data.mailer, null);
  const stranger = await signUp('nosy2', INVITE_PORT);
  assert.equal((await stranger.call('DELETE', '/admin/mailer')).status, 403);
});

test('with permission, the admin gets their Google contacts to complete an address, and nothing is kept', async () => {
  const boss = await theBoss();
  assert.equal((await googleFlow('google-boss-contacts:boss@gmail.com', { user: boss, mail: true })).result, 'mail');
  assert.deepEqual((await boss.call('GET', '/admin/overview')).data.mailer, { address: 'boss@gmail.com', contacts: true });
  const response = await fetch(`http://localhost:${INVITE_PORT}/api/admin/contacts`, { headers: { Cookie: boss.call.cookie() } });
  assert.match(response.headers.get('cache-control'), /no-store/);
  const { contacts, problem } = await response.json();
  assert.equal(problem, null);
  // Saved contacts and remembered ones together, each address once, a name where Google has one.
  assert.deepEqual([...contacts].sort((a, b) => a.email.localeCompare(b.email)), [
    { name: 'דנה כהן', email: 'dana.work@example.com' },
    { name: 'דנה כהן', email: 'dana@example.org' },
    { name: 'רון שכתבתי לו', email: 'ron@example.org' },
    { name: 'יוסי לוי', email: 'yossi@example.org' },
  ]);
  assert.ok(peopleRequests.length >= 3);
  for (const { pass } of peopleRequests) assert.equal(pass, 'Bearer pass-for-key-of-google-boss-contacts');
  const stranger = await signUp('nosy3', INVITE_PORT);
  assert.equal((await stranger.call('GET', '/admin/contacts')).status, 403);
  assert.equal((await client(INVITE_PORT)('GET', '/admin/contacts')).status, 401);

  // A Google project without the People API says so, and a withdrawn key asks to connect again.
  await googleFlow('google-boss-contacts-disabled:boss@gmail.com', { user: boss, mail: true });
  assert.deepEqual((await boss.call('GET', '/admin/contacts')).data, { contacts: null, problem: 'disabled' });
  await googleFlow('google-boss-contacts-withdrawn:boss@gmail.com', { user: boss, mail: true });
  assert.deepEqual((await boss.call('GET', '/admin/contacts')).data, { contacts: null, problem: 'expired' });
  await boss.call('DELETE', '/admin/mailer');
});

test('without Google settings the routes do not exist', async () => {
  const response = await fetch(`http://localhost:${PROD_PORT}/api/auth/google`, { redirect: 'manual' });
  assert.equal(response.status, 404);
  assert.equal((await client(PROD_PORT)('GET', '/me')).data.googleLogin, false);
});

test('a new login link returns a member to the same account', async () => {
  const boss = await theBoss();
  const { data: { members } } = await boss.call('GET', '/admin/overview');
  const known = members.find((user) => user.name === 'חברה');
  const { data } = await boss.call('POST', '/admin/invites', { userId: known.id });
  const { call } = await openInvite(data.link);
  assert.equal((await call('GET', '/me')).data.user.id, known.id);
  assert.equal((await boss.call('POST', '/admin/invites', { userId: 9999 })).status, 404);
});

test('only the admin reaches the admin screen', async () => {
  const stranger = await signUp('stranger', INVITE_PORT);
  assert.equal((await stranger.call('GET', '/admin/overview')).status, 403);
  assert.equal((await stranger.call('POST', '/admin/invites', {})).status, 403);
  const nobody = client(INVITE_PORT);
  assert.equal((await nobody('GET', '/admin/overview')).status, 401);
  assert.equal((await nobody('POST', '/admin/invites', {})).status, 401);
  assert.equal((await nobody('GET', '/me')).data.inviteOnly, true);
  assert.equal((await client(PROD_PORT)('GET', '/admin/overview')).status, 401);
});

test('the admin screen reports usage against the limits', async () => {
  const boss = await theBoss();
  const before = (await boss.call('GET', '/admin/overview')).data.usage;
  await upload(boss, phoneJpeg());
  const { usage, users } = (await boss.call('GET', '/admin/overview')).data;
  assert.equal(usage.today, before.today + 1);
  assert.equal(usage.bytes, before.bytes + 23);
  assert.equal(usage.storageLimit, 8 * 1024 ** 3);
  assert.equal(usage.uploadLimit, 2000);
  assert.ok(users >= 3);
});

test('the claims on /why are shown only to members, and the philosophy to everyone', async () => {
  const member = await signUp('reader');
  for (const path of ['/why', '/why.html']) {
    const response = await fetch(`http://localhost:${PORT}${path}`);
    assert.equal(response.status, 200, path);
    assert.match(response.headers.get('cache-control'), /no-store/, path);
    const html = await response.text();
    assert.match(html, /רשת חברתית היא דבר חשוב מדי/, path);
    assert.doesNotMatch(html, /data-unreviewed|איך זה בא לידי ביטוי|class="claim"/, path);
    const full = await (await fetch(`http://localhost:${PORT}${path}`, { headers: { Cookie: member.call.cookie() } })).text();
    assert.match(full, /רשת חברתית היא דבר חשוב מדי/, path);
    assert.match(full, /איך זה בא לידי ביטוי/, path);
    assert.match(full, /class="claim"/, path);
  }
  const stale = await fetch(`http://localhost:${PORT}/why`, { headers: { Cookie: 'session=nothing-real' } });
  assert.doesNotMatch(await stale.text(), /איך זה בא לידי ביטוי/);
});

test('the motto and the film page are public', async () => {
  assert.match(await (await fetch(`http://localhost:${PORT}/why`)).text(), /לבחור באופן דמוקרטי את העריץ שלך/);
  const film = await fetch(`http://localhost:${PORT}/intro`);
  assert.equal(film.status, 200);
  assert.match(await film.text(), /<video[^>]+src="\/video\/intro\.mp4"/);
});

let nava, oded, pnina, navaPost, odedComment, navaPhone;
const noticesOf = async (member, query = '') => (await member.call('GET', `/notices${query}`)).data;
const unseen = async (member) => (await member.call('GET', '/me')).data.user.notices;

test('a friend request and its acceptance are told, each to the other side', async () => {
  nava = await signUp('nava');
  oded = await signUp('oded');
  pnina = await signUp('pnina');
  await nava.call('POST', `/friends/${oded.id}`);
  let { notices } = await noticesOf(oded);
  assert.deepEqual(notices.map((item) => [item.kind, item.actor.name, item.text, item.path, item.fresh]), [['request', 'nava', 'בקשת חברות', '/friends', true]]);
  assert.equal(await unseen(oded), 1);
  assert.equal(await unseen(nava), 0);

  await oded.call('POST', `/friends/${nava.id}`);
  // A request that was answered is no longer a notice, and the one who asked hears the answer.
  assert.deepEqual((await noticesOf(oded)).notices, []);
  assert.equal(await unseen(oded), 0);
  ({ notices } = await noticesOf(nava));
  assert.deepEqual(notices.map((item) => [item.kind, item.actor.id, item.text, item.path]), [['accepted', oded.id, 'בקשת החברות שלך אושרה', `/u/${oded.id}`]]);
  await pnina.call('POST', `/friends/${nava.id}`);
  await nava.call('POST', `/friends/${pnina.id}`);
});

test('a comment, an answer and a word in a thread are told to whoever they concern, each on its own line', async () => {
  navaPost = (await nava.call('POST', '/posts', { body: 'הפוסט של נועה', audience: 'friends' })).data.id;
  odedComment = (await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'תגובה ראשונה' })).data.id;
  let [first] = (await noticesOf(nava)).notices;
  assert.deepEqual([first.kind, first.actor.name, first.text, first.words, first.about, first.path, first.fresh],
    ['comment', 'oded', 'תגובה לפוסט שלך', 'תגובה ראשונה', 'הפוסט של נועה', `/p/${navaPost}?c=${odedComment}`, true]);

  // Paz answers Omer: Omer hears an answer, Noa hears a comment on her post, each once.
  const pninaReply = (await pnina.call('POST', `/posts/${navaPost}/comments`, { body: 'תשובה לעומר', parentId: odedComment })).data.id;
  [first] = (await noticesOf(oded)).notices;
  assert.deepEqual([first.kind, first.text, first.words, first.about, first.path], ['reply', 'תשובה לתגובה שלך', 'תשובה לעומר', 'תגובה ראשונה', `/p/${navaPost}?c=${pninaReply}`]);
  assert.deepEqual((await noticesOf(nava)).notices.map((item) => [item.kind, item.actor.name]), [['comment', 'pnina'], ['comment', 'oded'], ['accepted', 'oded']]);

  // Omer writes again under his own comment: Paz, who wrote in that thread, hears of it.
  await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'ועוד מילה', parentId: odedComment });
  [first] = (await noticesOf(pnina)).notices;
  assert.deepEqual([first.kind, first.actor.name, first.text, first.words], ['thread', 'oded', 'תגובה נוספת בשרשור שכתבת בו', 'ועוד מילה']);
  // Noa answering in the thread under her own post is told to both, and never to herself.
  await nava.call('POST', `/posts/${navaPost}/comments`, { body: 'תודה לשניכם', parentId: odedComment });
  assert.equal((await noticesOf(oded)).notices[0].kind, 'reply');
  assert.equal((await noticesOf(pnina)).notices[0].kind, 'thread');
  assert.equal((await noticesOf(nava)).notices.filter((item) => item.actor.id === nava.id).length, 0);
  assert.equal(await unseen(nava), 4);
});

test('opening the notices marks them seen on every device, and what comes later is counted again', async () => {
  const { notices, now } = await noticesOf(nava);
  assert.ok(notices.every((item) => item.fresh));
  await nava.call('PUT', '/notices/seen', { upTo: now });
  assert.equal(await unseen(nava), 0);
  assert.ok((await noticesOf(nava)).notices.every((item) => !item.fresh));
  // A mark from the future is not believed.
  await nava.call('PUT', '/notices/seen', { upTo: Date.now() + 86400000 });
  await new Promise((resolve) => setTimeout(resolve, 5));
  await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'אחרי שראית' });
  assert.equal(await unseen(nava), 1);
  assert.deepEqual((await noticesOf(nava)).notices.map((item) => item.fresh).slice(0, 2), [true, false]);
});

test('a feeling is told once, follows the feeling that stands, and leaves when it is taken back', async () => {
  await nava.call('PUT', '/notices/seen', {});
  await oded.call('PUT', `/posts/${navaPost}/reaction`, { kind: 'love' });
  let felt = (await noticesOf(nava, '?group=feelings')).notices;
  assert.deepEqual(felt.map((item) => [item.kind, item.feeling, item.text, item.about, item.path]), [['felt', 'love', 'על הפוסט שלך', 'הפוסט של נועה', `/p/${navaPost}`]]);
  await oded.call('PUT', `/posts/${navaPost}/reaction`, { kind: '😂' });
  felt = (await noticesOf(nava, '?group=feelings')).notices;
  assert.deepEqual(felt.map((item) => item.feeling), ['funny']);
  assert.equal(await unseen(nava), 1);
  await oded.call('DELETE', `/posts/${navaPost}/reaction`);
  assert.deepEqual((await noticesOf(nava, '?group=feelings')).notices, []);
  assert.equal(await unseen(nava), 0);

  // A feeling about a comment is told to whoever wrote the comment, and leads to it.
  await nava.call('PUT', `/comments/${odedComment}/reaction`, { kind: '🙏' });
  const [mine] = (await noticesOf(oded, '?group=feelings')).notices;
  assert.deepEqual([mine.feeling, mine.text, mine.about, mine.path], ['🙏', 'על התגובה שלך', 'תגובה ראשונה', `/p/${navaPost}?c=${odedComment}`]);
});

test('a share is told to the writer, and a share of a share also to whoever it came through', async () => {
  const open = (await nava.call('POST', '/posts', { body: 'פוסט לכולם', audience: 'public' })).data.id;
  const odedShare = (await oded.call('POST', `/posts/${open}/share`)).data.id;
  let [told] = (await noticesOf(nava, '?group=shares')).notices;
  assert.deepEqual([told.kind, told.actor.name, told.text, told.about, told.path], ['share', 'oded', 'שיתוף של הפוסט שלך', 'פוסט לכולם', `/p/${odedShare}`]);

  // Rina is a friend of Omer only. She shares from his share: Noa hears of a share she cannot
  // open, so it leads to her own post, and Omer hears that it went on through him.
  const rina = await signUp('rina');
  await rina.call('POST', `/friends/${oded.id}`);
  await oded.call('POST', `/friends/${rina.id}`);
  const rinaShare = (await rina.call('POST', `/posts/${odedShare}/share`)).data.id;
  [told] = (await noticesOf(nava, '?group=shares')).notices;
  assert.deepEqual([told.kind, told.actor.name, told.path], ['share', 'rina', `/p/${open}`]);
  [told] = (await noticesOf(oded, '?group=shares')).notices;
  assert.deepEqual([told.kind, told.actor.name, told.text, told.path], ['reshare', 'rina', 'שיתוף נוסף, דרך השיתוף שלך', `/p/${rinaShare}`]);

  // Taking the share back takes the notices with it.
  await rina.call('DELETE', `/posts/${rinaShare}`);
  assert.deepEqual((await noticesOf(nava, '?group=shares')).notices.map((item) => item.actor.name), ['oded']);
  assert.deepEqual((await noticesOf(oded, '?group=shares')).notices, []);
});

test('a notice never outlives what it tells of, or the right to read it', async () => {
  const before = (await noticesOf(nava)).notices.length;
  const extra = (await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'תגובה שתוסר' })).data.id;
  assert.equal((await noticesOf(nava)).notices.length, before + 1);
  await oded.call('DELETE', `/comments/${extra}`);
  assert.equal((await noticesOf(nava)).notices.length, before);

  // Paz was told about words under Noa's post for friends. Once they are no longer friends she
  // cannot read the post, so those notices are not shown to her and are not counted.
  assert.ok((await noticesOf(pnina)).notices.some((item) => item.kind === 'thread'));
  assert.ok(await unseen(pnina) > 0);
  await nava.call('DELETE', `/friends/${pnina.id}`);
  assert.deepEqual((await noticesOf(pnina)).notices.filter((item) => item.path.startsWith(`/p/${navaPost}`)), []);
  assert.equal(await unseen(pnina), 0);
  // And nobody reads the notices of someone else.
  assert.equal((await anon('GET', '/notices')).status, 401);
});

test('the notices go back to the first one, page after page, with none skipped or repeated', async () => {
  const sara = await signUp('sara');
  const tomer = await signUp('tomer');
  await sara.call('POST', `/friends/${tomer.id}`);
  await tomer.call('POST', `/friends/${sara.id}`);
  const post = (await sara.call('POST', '/posts', { body: 'הרבה תגובות', audience: 'friends' })).data.id;
  for (let i = 1; i <= 34; i++) await tomer.call('POST', `/posts/${post}/comments`, { body: `תגובה ${i}` });
  const first = await noticesOf(sara, '?group=comments');
  assert.equal(first.notices.length, 30);
  assert.equal(first.notices[0].words, 'תגובה 34');
  const second = await noticesOf(sara, `?group=comments&before=${first.before}`);
  assert.deepEqual(second.notices.map((item) => item.words), ['תגובה 4', 'תגובה 3', 'תגובה 2', 'תגובה 1']);
  assert.equal(second.before, null);
  // Without the filter the same pages end with how the friendship began.
  const all = await noticesOf(sara, `?before=${first.before}`);
  assert.deepEqual(all.notices.map((item) => item.kind), ['comment', 'comment', 'comment', 'comment', 'accepted']);
});

test('the reading of a push here agrees with the example in the standard (RFC 8291)', () => {
  const ecdh = createECDH('prime256v1');
  ecdh.setPrivateKey(Buffer.from('q1dXpw3UpT5VOmu_cf_v6ih07Aems3njxI-JWgLcM94', 'base64url'));
  const body = Buffer.from('DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A_yl95bQpu6cVPTpK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQexSgSxsj_Qulcy4a-fN', 'base64url');
  assert.equal(openPush(ecdh, Buffer.from('BTBZMqHH6r4Tts7J_aSIgg', 'base64url'), body), 'When I grow up, I want to be a watermelon');
});

test('a push reaches the device that asked for one, sealed for it alone and signed by the network', async () => {
  const me = (await nava.call('GET', '/me')).data;
  assert.equal(me.pushKey, vapid.public);
  assert.deepEqual(me.user.pushMuted, []);
  navaPhone = device('nava-phone');
  const phone = navaPhone;
  // Only the address of a real push service is taken, so the network cannot be sent anywhere.
  for (const bad of [{ ...phone.subscription, endpoint: 'https://evil.example/push' }, { ...phone.subscription, keys: { p256dh: 'short', auth: 'x' } }, {}]) {
    assert.equal((await nava.call('PUT', '/push', bad)).status, 400);
  }
  assert.equal((await anon('PUT', '/push', phone.subscription)).status, 401);
  assert.equal((await nava.call('PUT', '/push', phone.subscription)).status, 200);
  assert.equal((await nava.call('PUT', '/push', phone.subscription)).status, 200);

  const written = (await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'תגובה עם התראה' })).data.id;
  const [push] = await phone.wait(1);
  assert.deepEqual([push.message.title, push.message.body, push.message.path], ['oded', 'תגובה לפוסט שלך: תגובה עם התראה', `/p/${navaPost}?c=${written}`]);
  assert.equal(push.message.count, await unseen(nava));
  assert.equal(push.headers['content-encoding'], 'aes128gcm');
  assert.ok(Number(push.headers.ttl) > 3600);
  // The words are not in what the push service was handed.
  assert.ok(!push.body.includes(Buffer.from('oded')));

  const [, token, key] = push.headers.authorization.match(/^vapid t=([\w.-]+), k=([\w-]+)$/);
  assert.equal(key, vapid.public);
  const [head, claims, signature] = token.split('.');
  assert.ok(verify('sha256', Buffer.from(`${head}.${claims}`), { key: createPublicKey({ key: vapid.jwk, format: 'jwk' }), dsaEncoding: 'ieee-p1363' }, Buffer.from(signature, 'base64url')));
  const said = JSON.parse(Buffer.from(claims, 'base64url'));
  assert.equal(said.aud, `http://localhost:${YOUTUBE_PORT}`);
  assert.equal(said.sub, `http://localhost:${PORT}`);
  assert.ok(said.exp > Date.now() / 1000 && said.exp <= Date.now() / 1000 + 86400);

  // A private message is pushed too, and leads into the conversation.
  await oded.call('POST', `/messages/${nava.id}`, { body: 'היי נועה' });
  const [, message] = await phone.wait(2);
  assert.deepEqual([message.message.title, message.message.body, message.message.path, message.message.tag], ['oded', 'היי נועה', `/messages/${oded.id}`, `m${oded.id}`]);
});

test('a feeling rings once, a silenced kind is not pushed, and a device that left is forgotten', async () => {
  const phone = navaPhone;
  const start = phone.count();
  const fresh = (await nava.call('POST', '/posts', { body: 'פוסט חדש', audience: 'friends' })).data.id;
  await oded.call('PUT', `/posts/${fresh}/reaction`, { kind: 'love' });
  const felt = (await phone.wait(start + 1)).at(-1);
  assert.deepEqual([felt.message.title, felt.message.body, felt.message.path], ['oded', '❤️ על הפוסט שלך', `/p/${fresh}`]);
  // Taking it back and giving it again, or changing it, does not ring again.
  await oded.call('DELETE', `/posts/${fresh}/reaction`);
  await oded.call('PUT', `/posts/${fresh}/reaction`, { kind: 'anger' });

  assert.deepEqual((await nava.call('PUT', '/push/muted', { muted: ['messages', 'nonsense'] })).data.muted, ['messages']);
  assert.deepEqual((await nava.call('GET', '/me')).data.user.pushMuted, ['messages']);
  await oded.call('POST', `/messages/${nava.id}`, { body: 'בשקט' });
  await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'בקול' });
  const last = (await phone.wait(start + 2)).at(-1);
  assert.equal(last.message.body, 'תגובה לפוסט שלך: בקול');
  await nava.call('PUT', '/push/muted', { muted: [] });

  // A second device that has unsubscribed answers 410, and is not tried again.
  const old = device('gone');
  await nava.call('PUT', '/push', old.subscription);
  assert.equal((await nava.call('POST', '/push/test')).data.delivered, 1);
  await old.wait(1);
  assert.equal((await nava.call('POST', '/push/test')).data.delivered, 1);
  assert.equal(old.count(), 1);
  const test = (await phone.wait(start + 4)).at(-1);
  assert.equal(test.message.path, '/notices');
});

test('logging out silences the device, and it is told again only after it asks again', async () => {
  const phone = navaPhone;
  const start = phone.count();
  await nava.call('POST', '/auth/logout');
  await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'אחרי היציאה' });
  await new Promise((resolve) => setTimeout(resolve, 600));
  assert.equal(phone.count(), start);

  const { data } = await nava.call('POST', '/auth/request', { email: 'nava@example.com' });
  await nava.call('POST', '/auth/verify', { token: data.devLink.split('#token=')[1] });
  // The comment waited on the page all along.
  assert.equal((await noticesOf(nava)).notices[0].words, 'אחרי היציאה');
  await nava.call('PUT', '/push', phone.subscription);
  await nava.call('DELETE', '/push', { endpoint: phone.subscription.endpoint });
  await oded.call('POST', `/posts/${navaPost}/comments`, { body: 'אחרי הכיבוי' });
  await new Promise((resolve) => setTimeout(resolve, 600));
  assert.equal(phone.count(), start);
});

test('a tagged friend becomes a link and is told, and only a friend whose name stands in the words is tagged', async () => {
  const [writer, friend, other, stranger] = await Promise.all(['tag-writer', 'tag-friend', 'tag-other', 'tag-stranger'].map((name) => signUp(name)));
  const befriend = async (x, y) => {
    await x.call('POST', `/friends/${y.id}`);
    await y.call('POST', `/friends/${x.id}`);
  };
  await befriend(writer, friend);
  await befriend(writer, other);
  const tagsOf = async (reader, id) => (await reader.call('GET', `/posts/${id}`)).data.post.mentions;
  const latest = async (member) => (await noticesOf(member)).notices[0];

  // Three are named: a friend whose name is written, a friend whose name is not, and a stranger.
  const { id } = (await writer.call('POST', '/posts', {
    body: 'תודה ל-@tag-friend על העזרה, וגם ל-@tag-stranger', audience: 'friends', mentions: [friend.id, other.id, stranger.id, 'x', 999999],
  })).data;
  assert.deepEqual(await tagsOf(friend, id), [{ id: friend.id, name: 'tag-friend' }]);
  const told = await latest(friend);
  assert.deepEqual([told.kind, told.actor.name, told.text, told.words, told.about, told.path],
    ['mention', 'tag-writer', 'תיוג שלך בפוסט', 'תודה ל-@tag-friend על העזרה, וגם ל-@tag-stranger', null, `/p/${id}`]);
  assert.ok(!(await noticesOf(other)).notices.some((item) => item.kind === 'mention'));
  assert.equal((await noticesOf(stranger)).notices.length, 0);
  assert.deepEqual((await noticesOf(friend, '?group=mentions')).notices.map((item) => item.kind), ['mention']);

  // Rewriting: a tag that was added is told, one that stayed is not told again, one that left is gone.
  let { post } = (await writer.call('PUT', `/posts/${id}`, { body: 'תודה ל-@tag-friend ול-@tag-other', mentions: [friend.id, other.id] })).data;
  assert.deepEqual(post.mentions.map((tag) => tag.name), ['tag-friend', 'tag-other']);
  assert.equal((await latest(other)).kind, 'mention');
  assert.equal((await noticesOf(friend)).notices.filter((item) => item.kind === 'mention').length, 1);
  // An older browser that sends no tags keeps those still written.
  ({ post } = (await writer.call('PUT', `/posts/${id}`, { body: 'תודה רק ל-@tag-other' })).data);
  assert.deepEqual(post.mentions, [{ id: other.id, name: 'tag-other' }]);
  assert.ok(!(await noticesOf(friend)).notices.some((item) => item.kind === 'mention'));
  // The tag keeps the name as it was written, also after the friend changed theirs.
  await other.call('PUT', '/me', { name: 'tag-renamed' });
  assert.deepEqual(await tagsOf(writer, id), [{ id: other.id, name: 'tag-other' }]);
  ({ post } = (await writer.call('PUT', `/posts/${id}`, { body: 'שוב תודה ל-@tag-other', mentions: [other.id] })).data);
  assert.deepEqual(post.mentions, [{ id: other.id, name: 'tag-other' }]);
});

test('a tag in a comment is told once, and never about a post the tagged friend may not read', async () => {
  const [owner, guest, pal] = await Promise.all(['tagc-owner', 'tagc-guest', 'tagc-pal'].map((name) => signUp(name)));
  const befriend = async (x, y) => {
    await x.call('POST', `/friends/${y.id}`);
    await y.call('POST', `/friends/${x.id}`);
  };
  await befriend(owner, guest);
  await befriend(guest, pal);
  // Pal is a friend of the guest and not of the owner: a closed post stays closed to them.
  const closed = (await owner.call('POST', '/posts', { body: 'רק לחברים', audience: 'friends' })).data.id;
  await guest.call('POST', `/posts/${closed}/comments`, { body: 'מה דעתך @tagc-pal?', mentions: [pal.id] });
  assert.equal((await noticesOf(pal)).notices.filter((item) => item.kind === 'mention').length, 0);

  const open = (await owner.call('POST', '/posts', { body: 'פתוח לכולם', audience: 'public' })).data.id;
  const comment = (await guest.call('POST', `/posts/${open}/comments`, { body: 'מה דעתך @tagc-pal? וגם @tagc-owner', mentions: [pal.id, owner.id] })).data.id;
  const told = (await noticesOf(pal)).notices[0];
  assert.deepEqual([told.kind, told.text, told.words, told.about, told.path],
    ['mention', 'תיוג שלך בתגובה', 'מה דעתך @tagc-pal? וגם @tagc-owner', 'פתוח לכולם', `/p/${open}?c=${comment}`]);
  // The owner already hears of the comment on their post, so the tag adds no second line.
  assert.deepEqual((await noticesOf(owner)).notices.filter((item) => item.post !== null && item.path.includes(`c=${comment}`)).map((item) => item.kind), ['comment']);
  const shown = (await pal.call('GET', `/posts/${open}`)).data.comments.find((item) => item.id === comment);
  assert.deepEqual(shown.mentions.map((tag) => tag.id), [pal.id, owner.id]);
  // Rewriting the comment without the tag takes the notice away, and removing the comment clears its tags.
  await guest.call('PUT', `/comments/${comment}`, { body: 'בלי תיוג', mentions: [] });
  assert.equal((await noticesOf(pal)).notices.filter((item) => item.kind === 'mention').length, 0);
});

test('friends see the friends of a friend, and a stranger does not', async () => {
  const [a, b, c2, d] = await Promise.all(['fof-a', 'fof-b', 'fof-c', 'fof-d'].map((name) => signUp(name)));
  const befriend = async (x, y) => {
    await x.call('POST', `/friends/${y.id}`);
    await y.call('POST', `/friends/${x.id}`);
  };
  // a - b, b - c, b - d, c - d. So a knows c and d only through b.
  await befriend(a, b);
  await befriend(b, c2);
  await befriend(b, d);
  await befriend(c2, d);
  const { data } = await a.call('GET', `/users/${b.id}/friends`);
  assert.deepEqual(data.friends.map((f) => [f.name, f.friendship, f.mutual]), [['fof-a', 'self', 0], ['fof-c', null, 1], ['fof-d', null, 1]]);
  assert.equal((await a.call('GET', `/users/${b.id}`)).data.friends, 3);
  // c is not a friend of a: a sees neither the list nor the number, and nobody sees it logged out.
  assert.equal((await a.call('GET', `/users/${c2.id}/friends`)).status, 403);
  assert.equal((await a.call('GET', `/users/${c2.id}`)).data.friends, null);
  assert.equal((await anon('GET', `/users/${b.id}/friends`)).status, 401);
  // Everyone sees their own list, with what each friend is to them.
  assert.deepEqual((await c2.call('GET', `/users/${c2.id}/friends`)).data.friends.map((f) => [f.name, f.friendship, f.mutual]), [['fof-b', 'friends', 1], ['fof-d', 'friends', 1]]);

  // The suggestions name exactly the friends in common, and leave once a request was sent.
  let { people } = (await a.call('GET', '/friends/suggested')).data;
  assert.deepEqual(people.map((one) => [one.name, one.through.map((m) => m.name)]), [['fof-c', ['fof-b']], ['fof-d', ['fof-b']]]);
  await a.call('POST', `/friends/${c2.id}`);
  assert.deepEqual((await a.call('GET', `/users/${b.id}/friends`)).data.friends.map((f) => f.friendship), ['self', 'outgoing', null]);
  assert.deepEqual((await c2.call('GET', `/users/${b.id}/friends`)).data.friends.map((f) => [f.name, f.friendship]), [['fof-a', 'incoming'], ['fof-c', 'self'], ['fof-d', 'friends']]);
  ({ people } = (await a.call('GET', '/friends/suggested')).data);
  assert.deepEqual(people.map((one) => one.name), ['fof-d']);
  // Whoever shares more friends comes first: after a and c are friends, d is known through two.
  await c2.call('POST', `/friends/${a.id}`);
  ({ people } = (await a.call('GET', '/friends/suggested')).data);
  assert.deepEqual(people.map((one) => [one.name, one.through.map((m) => m.name).sort()]), [['fof-d', ['fof-b', 'fof-c']]]);
});

test('without DEV_LOGIN a login link is never returned', async () => {
  const { status, data } = await client(PROD_PORT)('POST', '/auth/request', { email: 'y@example.com' });
  assert.equal(status, 503);
  assert.equal(data.devLink, undefined);
});

// A run cut short must not leave servers holding the ports for the next one.
function stopServers() {
  for (const server of servers) {
    try {
      process.kill(-server.pid);
    } catch {
      // Already gone.
    }
  }
}
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    stopServers();
    process.exit(1);
  });
}

let failed = 0;
try {
  await new Promise((resolve) => fakeYoutube.listen(YOUTUBE_PORT, resolve));
  const fake = `http://localhost:${YOUTUBE_PORT}`;
  startServer(PORT, ['--var', `VAPID_PUBLIC:${vapid.public}`, '--var', `VAPID_PRIVATE:${vapid.private}`, '--var', 'DEV_LOGIN:1', '--var', `YOUTUBE_ORIGIN:${fake}`, '--var', `TILES_ORIGIN:${fake}`, '--var', `WIKI_HE_ORIGIN:${fake}`,
    '--var', `WIKI_EN_ORIGIN:${fake}`, '--var', `PARDESPEDIA_ORIGIN:${fake}`, '--var', `PLANS_URL:${fake}/plans.json`,
    ...['N12', 'C13', 'C14', 'KAN'].flatMap((name) => ['--var', `${name}_FEED:${fake}/feed/${name.toLowerCase()}.xml`])]);
  startServer(PROD_PORT, []);
  startServer(STORAGE_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'STORAGE_LIMIT_BYTES:30']);
  startServer(UPLOADS_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'UPLOADS_PER_DAY:2']);
  startServer(INVITE_PORT, ['--var', `PEOPLE_ORIGIN:http://localhost:${YOUTUBE_PORT}`, '--var', 'DEV_LOGIN:1', '--var', 'INVITE_ONLY:1', '--var', 'ADMIN_EMAIL:boss@example.com',
    '--var', 'GOOGLE_CLIENT_ID:test-client', '--var', 'GOOGLE_CLIENT_SECRET:test-secret',
    '--var', `GOOGLE_TOKEN_URL:http://localhost:${YOUTUBE_PORT}/token`, '--var', `GOOGLE_PICTURE_ORIGIN:http://localhost:${YOUTUBE_PORT}`,
    '--var', `GMAIL_ORIGIN:http://localhost:${YOUTUBE_PORT}`, '--var', `GOOGLE_REVOKE_URL:http://localhost:${YOUTUBE_PORT}/revoke`]);
  await Promise.all([PORT, PROD_PORT, STORAGE_PORT, UPLOADS_PORT, INVITE_PORT].map(waitFor));
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log(`ok   ${name}`);
    } catch (error) {
      failed++;
      console.log(`FAIL ${name}\n     ${error.message}`);
    }
  }
} finally {
  stopServers();
  fakeYoutube.close();
  rmSync(root, { recursive: true, force: true });
}
console.log(failed ? `${failed} failed` : `all ${tests.length} passed`);
process.exit(failed ? 1 : 0);
