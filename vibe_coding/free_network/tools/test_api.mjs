// Runs the real worker on a throwaway local database and checks the promises the network makes.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Other projects' servers sometimes sit on these ports. TEST_PORT_BASE=18792 npm test moves them all.
const BASE = Number(process.env.TEST_PORT_BASE ?? 8792);
const PORT = BASE + 7;
const PROD_PORT = BASE + 6;
const DEMO_PORT = BASE + 4;
const STORAGE_PORT = BASE + 3;
const UPLOADS_PORT = BASE + 2;
const PILOT_PORT = BASE + 1;
const YOUTUBE_PORT = BASE;
const VIDEO_ID = 'abcdefghijk';
const youtubeRequests = [];
const digestRequests = [];
const pictureRequests = [];
const mailSent = [];
const revoked = [];

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
      const scope = `openid email${code.includes('unticked') ? '' : ' https://www.googleapis.com/auth/gmail.send'}`;
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
function startServer(port, extraArgs, { database = 'free-network', seed } = {}) {
  const state = join(root, String(port));
  const envArgs = extraArgs.filter((arg) => arg.startsWith('--env') || arg === 'demo');
  const quiet = { stdio: 'ignore', env: { ...process.env, CI: '1' } };
  execFileSync('npx', ['wrangler', 'd1', 'migrations', 'apply', database, '--local', '--persist-to', state, ...envArgs], quiet);
  if (seed) execFileSync('npx', ['wrangler', 'd1', 'execute', database, '--local', '--persist-to', state, ...envArgs, '--file', seed], quiet);
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

// One admin for all pilot tests: logins are rate limited per address, as in production.
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
  assert.equal((await write(dana, eli, 'x'.repeat(4001))).status, 400);
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

test('the demo globe has friends on the earth and friends on the cloud', async () => {
  const call = client(DEMO_PORT);
  await call('POST', '/auth/demo');
  const { friends } = (await call('GET', '/globe')).data;
  assert.equal(friends.length, 8);
  assert.equal(friends.filter((friend) => !friend.place).length, 2);
});

async function theBoss() {
  bossAccount ??= await signUp('boss', PILOT_PORT);
  return bossAccount;
}

async function openInvite(link) {
  const call = client(PILOT_PORT);
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
async function googleFlow(code, { user, link = false, mail = false, forgedState = false } = {}) {
  const base = `http://localhost:${PILOT_PORT}/api/auth/google`;
  const session = user?.call.cookie() ?? '';
  const start = await fetch(`${base}${link ? '?link=1' : mail ? '?mail=1' : ''}`, { redirect: 'manual', headers: { Cookie: session } });
  const asked = new URL(start.headers.get('location'), base);
  const state = asked.searchParams.get('state');
  if (!state) return { result: null, asked };
  const stateCookie = start.headers.get('set-cookie')?.split(';')[0] ?? '';
  const back = await fetch(`${base}/callback?code=${code}&state=${forgedState ? 'forged' : state}`, {
    redirect: 'manual',
    headers: { Cookie: [session, stateCookie].filter(Boolean).join('; ') },
  });
  const cookie = back.headers.getSetCookie().find((line) => line.startsWith('session='))?.split(';')[0];
  return { result: new URL(back.headers.get('location'), base).searchParams.get('google'), cookie, asked };
}

test('a linked Google account opens the same account on another device', async () => {
  const lior = await signUp('lior', PILOT_PORT);
  assert.equal((await googleFlow('google-lior', { user: lior, link: true })).result, 'linked');
  assert.equal((await lior.call('GET', '/me')).data.user.googleLinked, true);
  const { result, cookie } = await googleFlow('google-lior');
  assert.equal(result, 'login');
  const me = await fetch(`http://localhost:${PILOT_PORT}/api/me`, { headers: { Cookie: cookie } });
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

test('one Google account belongs to one person, and it can be unlinked', async () => {
  const maya = await signUp('maya', PILOT_PORT);
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
  const me = await (await fetch(`http://localhost:${PILOT_PORT}/api/me`, { headers })).json();
  assert.equal(me.user.googleLinked, true);
  assert.equal(me.user.name, 'חבר חדש');
  const picture = await fetch(`http://localhost:${PILOT_PORT}/api/images/${me.user.avatar}`);
  assert.equal(picture.status, 200);
  const stored = Buffer.from(await picture.arrayBuffer());
  assert.ok(!stored.includes('GPS'));
  assert.deepEqual(pictureRequests, ['/a-/photo=s400-c']);
  const friends = await (await fetch(`http://localhost:${PILOT_PORT}/api/friends`, { headers })).json();
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
  const me = await (await fetch(`http://localhost:${PILOT_PORT}/api/me`, { headers: { Cookie: back.cookie } })).json();
  assert.equal(me.user.id, id);
  assert.equal(me.user.googleLinked, true);
  // Google's drawn letter is not a photo, so the member keeps the picture made of their name.
  assert.equal(me.user.avatar, null);
});

test('linking Google brings its picture only to someone who has none', async () => {
  const noPicture = await signUp('pictureless', PILOT_PORT);
  assert.equal((await googleFlow('google-pictureless:p@gmail.com:photo', { user: noPicture, link: true })).result, 'linked');
  assert.ok((await noPicture.call('GET', '/me')).data.user.avatar);
  const chosen = await signUp('chosen', PILOT_PORT);
  const own = (await upload(chosen, phoneJpeg())).data.id;
  await chosen.call('PUT', '/me/images', { kind: 'avatar', imageId: own });
  await googleFlow('google-chosen:c@gmail.com:photo', { user: chosen, link: true });
  assert.equal((await chosen.call('GET', '/me')).data.user.avatar, own);
  const elsewhere = await signUp('elsewhere', PILOT_PORT);
  await googleFlow('google-elsewhere:e@gmail.com:elsewhere', { user: elsewhere, link: true });
  assert.equal((await elsewhere.call('GET', '/me')).data.user.avatar, null);
});

test('an email invitation needs a verified address, and only the admin can send or cancel one', async () => {
  const boss = await theBoss();
  await boss.call('POST', '/admin/email-invites', { email: 'careful@gmail.com' });
  assert.equal((await googleFlow('google-careful:careful@gmail.com!')).result, 'unknown');
  const stranger = await signUp('stranger', PILOT_PORT);
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
  const stranger = await signUp('nosy', PILOT_PORT);
  assert.equal((await googleFlow('google-nosy:nosy@gmail.com', { user: stranger, mail: true })).result, null);
  const unticked = await googleFlow('google-boss-unticked:boss@gmail.com', { user: boss, mail: true });
  assert.equal(unticked.result, 'mail-refused');
  assert.equal((await boss.call('GET', '/admin/overview')).data.mailer, null);
  const connected = await googleFlow('google-boss:The.Boss@gmail.com', { user: boss, mail: true });
  assert.equal(connected.result, 'mail');
  assert.equal(connected.asked.searchParams.get('scope'), 'openid email https://www.googleapis.com/auth/gmail.send');
  assert.equal(connected.asked.searchParams.get('access_type'), 'offline');
  assert.deepEqual((await boss.call('GET', '/admin/overview')).data.mailer, { address: 'The.Boss@gmail.com' });
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
  const stranger = await signUp('nosy2', PILOT_PORT);
  assert.equal((await stranger.call('DELETE', '/admin/mailer')).status, 403);
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
  const stranger = await signUp('stranger', PILOT_PORT);
  assert.equal((await stranger.call('GET', '/admin/overview')).status, 403);
  assert.equal((await stranger.call('POST', '/admin/invites', {})).status, 403);
  const nobody = client(PILOT_PORT);
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

test('the demo opens with one click and shows the invented feed', async () => {
  const call = client(DEMO_PORT);
  assert.equal((await call('GET', '/me')).data.demo, true);
  assert.equal((await call('POST', '/auth/demo')).status, 200);
  assert.equal((await call('GET', '/me')).data.user.name, 'אורח בדמו');
  assert.equal((await call('GET', '/feed')).data.posts.length, 4);
});

test('the demo refuses every write and stores nothing', async () => {
  const call = client(DEMO_PORT);
  await call('POST', '/auth/demo');
  const writes = [
    ['POST', '/posts', { body: 'x', audience: 'public' }],
    ['POST', '/posts/3/comments', { body: 'x' }],
    ['PUT', '/me', { name: 'x' }],
    ['POST', '/friends/5'],
    ['DELETE', '/friends/2'],
    ['POST', '/auth/request', { email: 'x@example.com' }],
    ['POST', '/client-log', { message: 'x' }],
    ['PUT', '/me/place', { lat: 1, lng: 1 }],
    ['DELETE', '/me/place'],
    ['POST', '/messages/2', { body: 'x' }],
    ['PUT', '/messages/2/read', {}],
    ['DELETE', '/messages/2/2'],
  ];
  for (const [method, path, body] of writes) assert.equal((await call(method, path, body)).status, 403, path);
  assert.equal((await call('GET', '/feed')).data.posts.length, 4);
  assert.equal((await call('GET', '/friends')).data.friends.length, 8);
});

test('the demo shows invented conversations, all of them read', async () => {
  const call = client(DEMO_PORT);
  await call('POST', '/auth/demo');
  assert.deepEqual((await call('GET', '/messages')).data.conversations.map((item) => item.user.id), [3, 2]);
  assert.equal((await call('GET', '/me')).data.user.unreadMessages, 0);
  assert.deepEqual((await call('GET', '/messages/3')).data.messages.map((item) => item.removed), [false, true, false]);
  assert.equal((await call('GET', '/messages/2')).data.messages.length, 5);
});

test('logging out of the demo leaves it open for the next visitor', async () => {
  const first = client(DEMO_PORT);
  await first('POST', '/auth/demo');
  await first('POST', '/auth/logout');
  const second = client(DEMO_PORT);
  await second('POST', '/auth/demo');
  assert.equal((await second('GET', '/me')).data.user.id, 1);
});

test('the demo serves /why without the unreviewed claims, and demo login exists only there', async () => {
  for (const path of ['/why', '/why.html']) {
    const response = await fetch(`http://localhost:${DEMO_PORT}${path}`);
    assert.equal(response.status, 200, path);
    const html = await response.text();
    assert.match(html, /רשת חברתית היא דבר חשוב מדי/, path);
    assert.doesNotMatch(html, /data-unreviewed|איך זה בא לידי ביטוי|class="claim"/, path);
  }
  const full = await (await fetch(`http://localhost:${PORT}/why`)).text();
  assert.match(full, /איך זה בא לידי ביטוי/);
  assert.equal((await client(PROD_PORT)('POST', '/auth/demo')).status, 404);
});

test('the motto and the film page are public, in the network and in the demo', async () => {
  for (const port of [PORT, DEMO_PORT]) {
    assert.match(await (await fetch(`http://localhost:${port}/why`)).text(), /לבחור באופן דמוקרטי את העריץ שלך/);
    const film = await fetch(`http://localhost:${port}/intro`);
    assert.equal(film.status, 200);
    assert.match(await film.text(), /<video[^>]+src="\/video\/intro\.mp4"/);
  }
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
  startServer(PORT, ['--var', 'DEV_LOGIN:1', '--var', `YOUTUBE_ORIGIN:${fake}`, '--var', `WIKI_HE_ORIGIN:${fake}`,
    '--var', `WIKI_EN_ORIGIN:${fake}`, '--var', `PARDESPEDIA_ORIGIN:${fake}`, '--var', `PLANS_URL:${fake}/plans.json`,
    ...['N12', 'C13', 'C14', 'KAN'].flatMap((name) => ['--var', `${name}_FEED:${fake}/feed/${name.toLowerCase()}.xml`])]);
  startServer(PROD_PORT, []);
  startServer(DEMO_PORT, ['--env', 'demo'], { database: 'free-network-demo', seed: 'tools/demo_seed.sql' });
  startServer(STORAGE_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'STORAGE_LIMIT_BYTES:30']);
  startServer(UPLOADS_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'UPLOADS_PER_DAY:2']);
  startServer(PILOT_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'INVITE_ONLY:1', '--var', 'ADMIN_EMAIL:boss@example.com',
    '--var', 'GOOGLE_CLIENT_ID:test-client', '--var', 'GOOGLE_CLIENT_SECRET:test-secret',
    '--var', `GOOGLE_TOKEN_URL:http://localhost:${YOUTUBE_PORT}/token`, '--var', `GOOGLE_PICTURE_ORIGIN:http://localhost:${YOUTUBE_PORT}`,
    '--var', `GMAIL_ORIGIN:http://localhost:${YOUTUBE_PORT}`, '--var', `GOOGLE_REVOKE_URL:http://localhost:${YOUTUBE_PORT}/revoke`]);
  await Promise.all([PORT, PROD_PORT, DEMO_PORT, STORAGE_PORT, UPLOADS_PORT, PILOT_PORT].map(waitFor));
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
