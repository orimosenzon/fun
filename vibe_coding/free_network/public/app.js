const app = document.getElementById('app');
const nav = document.getElementById('nav');
const MAX_IMAGE_SIDE = 1600;
const MAX_IMAGES = 4;
const MAX_REPORTS = 20;
let reports = 0;
let me = null;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

// Sends browser-side failures to the server log, so they can be read without asking the user.
// The address hash is left out because a login link keeps its token there.
function report(message, stack) {
  if (reports >= MAX_REPORTS) return;
  reports++;
  fetch('/api/client-log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: String(message), stack, path: location.pathname }),
    keepalive: true,
  }).catch(() => {});
}

window.addEventListener('error', (e) => report(e.message, e.error?.stack));
window.addEventListener('unhandledrejection', (e) => report(e.reason?.message ?? e.reason, e.reason?.stack));

async function api(path, { method = 'GET', body } = {}) {
  const isFile = body instanceof Blob;
  let response;
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers: body ? { 'Content-Type': isFile ? body.type : 'application/json' } : {},
      body: isFile ? body : body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    report(`${method} ${path}: ${err.message}`);
    throw new Error('אין חיבור לשרת, נסו שוב');
  }
  const data = await response.json().catch(() => ({}));
  if (response.ok) return data;
  if (response.status < 500) throw new Error(data.error ?? 'הבקשה נדחתה');

  const code = data.requestId ?? response.headers.get('X-Request-Id');
  if (!data.requestId) report(`${method} ${path}: ${response.status} without a server log`);
  throw new Error(`${data.error ?? 'משהו השתבש, נסו שוב'}${code ? ` (קוד ${code})` : ''}`);
}

function go(path) {
  history.pushState(null, '', path);
  route();
}

function link(path, text) {
  return el('a', { href: path, onclick: (e) => { e.preventDefault(); go(path); } }, text);
}

// Text is added as text nodes only, so user content can never become HTML.
// Links are clickable for everyone, in posts and in comments alike.
function richText(text) {
  const node = el('p', { className: 'text', dir: 'auto' });
  let last = 0;
  for (const match of text.matchAll(/https?:\/\/[^\s<>"]+/g)) {
    const url = match[0].replace(/[.,;:!?'"”׳״)\]]+$/, '');
    node.append(text.slice(last, match.index));
    node.append(el('a', { href: url, target: '_blank', rel: 'noopener nofollow ugc' }, url));
    last = match.index + url.length;
  }
  node.append(text.slice(last));
  return node;
}

// Nothing is requested from Google, not even a thumbnail, until the reader clicks.
function video(id) {
  const button = el('button', { className: 'video', type: 'button' }, '▶ סרטון מיוטיוב, לחצו לטעינה');
  button.addEventListener('click', () => {
    button.replaceWith(el('div', { className: 'video' }, el('iframe', {
      src: `https://www.youtube-nocookie.com/embed/${id}?autoplay=1`,
      title: 'סרטון מיוטיוב',
      allow: 'autoplay; encrypted-media; picture-in-picture; fullscreen',
      allowFullscreen: true,
      referrerPolicy: 'strict-origin-when-cross-origin',
    })));
  });
  return button;
}

// Shrinks the photo and re-encodes it, which also drops the GPS location the phone stored in it.
async function prepareImage(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = el('canvas', { width: Math.round(bitmap.width * scale), height: Math.round(bitmap.height * scale) });
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.8));
}

function gallery(ids) {
  return el('div', { className: `gallery n${ids.length}` }, ...ids.map((id) => el('a', { href: `/api/images/${id}`, target: '_blank' },
    el('img', { src: `/api/images/${id}`, alt: 'תמונה מהפוסט', loading: 'lazy' }))));
}

function when(ms) {
  return new Intl.DateTimeFormat('he-IL', { dateStyle: 'short', timeStyle: 'short' }).format(ms);
}

function postCard(post, { onDeleted } = {}) {
  const url = `/p/${post.id}`;
  const count = post.commentCount === 1 ? 'תגובה אחת' : `${post.commentCount} תגובות`;
  const actions = el('div', { className: 'row spread' }, link(url, post.commentCount ? count : 'להגיב'));
  if (post.audience === 'public') {
    const copy = el('button', { className: 'plain', type: 'button' }, 'העתקת קישור');
    copy.addEventListener('click', async () => {
      await navigator.clipboard.writeText(new URL(url, location.origin).href);
      copy.textContent = 'הועתק';
    });
    actions.append(copy);
  }
  if (post.mine) {
    actions.append(el('button', {
      className: 'plain danger',
      type: 'button',
      onclick: async () => {
        if (!confirm('למחוק את הפוסט?')) return;
        await api(`/posts/${post.id}`, { method: 'DELETE' });
        onDeleted?.();
      },
    }, 'מחיקה'));
  }
  return el('article', { className: 'card' },
    el('div', { className: 'meta' },
      el('span', { className: 'author', dir: 'auto' }, post.author.name),
      el('span', { className: 'muted' }, `${when(post.createdAt)} · ${post.audience === 'public' ? 'ציבורי' : 'חברים'}`)),
    post.body ? richText(post.body) : null,
    post.images.length ? gallery(post.images) : null,
    post.youtubeId ? video(post.youtubeId) : null,
    actions);
}

async function composer(onPosted) {
  const { friends } = await api('/audience');
  const text = el('textarea', { placeholder: 'מה חדש?', maxLength: 5000, dir: 'auto' });
  const note = el('p', { className: 'muted' });
  const error = el('p', { className: 'error', hidden: true });
  const radio = (value, label, checked) => el('label', {},
    el('input', { type: 'radio', name: 'audience', value, checked, onchange: explain }), label);
  const choices = el('div', { className: 'audience' }, radio('friends', 'חברים', true), radio('public', 'ציבורי', false));

  function audience() {
    return choices.querySelector('input:checked').value;
  }

  function explain() {
    const who = friends === 1 ? 'החבר היחיד שלך' : `${friends} החברים שלך`;
    const reach = friends ? `יופיע בפיד של ${who}, בלי סינון ובלי דירוג.` : 'עוד אין לך חברים, אז הפוסט יופיע רק אצלך.';
    note.textContent = audience() === 'public' ? `${reach} כל מי שיש לו את הקישור יוכל לקרוא אותו.` : reach;
  }
  explain();

  let photos = [];
  const previews = el('div', { className: 'previews' });
  const showPhotos = () => previews.replaceChildren(...photos.map((photo) => el('button', {
    className: 'preview',
    type: 'button',
    title: 'הסרת התמונה',
    onclick: () => {
      photos = photos.filter((other) => other !== photo);
      showPhotos();
    },
  }, el('img', { src: URL.createObjectURL(photo), alt: 'תמונה שנבחרה' }))));
  const picker = el('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true });
  picker.addEventListener('change', async () => {
    error.hidden = true;
    try {
      for (const file of [...picker.files].slice(0, MAX_IMAGES - photos.length)) photos.push(await prepareImage(file));
      showPhotos();
    } catch {
      error.textContent = 'אי אפשר לקרוא את התמונה הזו';
      error.hidden = false;
    }
    picker.value = '';
  });
  const addPhoto = el('button', { className: 'plain', type: 'button', onclick: () => picker.click() }, 'הוספת תמונה');

  const submit = el('button', { type: 'submit' }, 'פרסום');
  return el('form', {
    className: 'card',
    onsubmit: async (e) => {
      e.preventDefault();
      submit.disabled = true;
      error.hidden = true;
      try {
        const uploads = photos.map((photo) => api('/images', { method: 'POST', body: photo }));
        const images = (await Promise.all(uploads)).map((upload) => upload.id);
        await api('/posts', { method: 'POST', body: { body: text.value, audience: audience(), images } });
        onPosted();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
        submit.disabled = false;
      }
    },
  }, text, previews, picker, el('div', { className: 'row spread' }, choices, addPhoto, submit), note, error);
}

async function feedPage() {
  const list = el('div');
  const more = el('button', { className: 'plain', type: 'button', hidden: true }, 'פוסטים ישנים יותר');
  let oldest = null;

  async function load() {
    const { posts, hasMore } = await api(`/feed${oldest ? `?before=${oldest}` : ''}`);
    list.append(...posts.map((post) => postCard(post, { onDeleted: route })));
    if (posts.length) oldest = posts.at(-1).id;
    more.hidden = !hasMore;
    if (!list.children.length) {
      list.append(el('p', { className: 'muted' }, 'עוד אין פוסטים. כדאי להתחיל בלמצוא חברים.'));
    }
  }
  more.addEventListener('click', load);
  await load();

  app.replaceChildren(
    await composer(route),
    el('p', { className: 'muted' }, 'הפיד כולל את כל הפוסטים שלך ושל החברים שלך, מהחדש לישן. זה כל האלגוריתם.'),
    list,
    more);
}

function comment(item, reload) {
  const node = el('div', { className: 'comment' },
    el('div', { className: 'meta' },
      el('span', { className: 'author', dir: 'auto' }, item.author.name),
      el('span', { className: 'muted' }, when(item.createdAt))));
  if (item.removedBy) {
    const by = item.removedBy === 'author' ? 'הכותב' : 'בעל הפוסט';
    node.append(el('p', { className: 'text removed' }, `התגובה הוסרה על ידי ${by}`));
    return node;
  }
  node.append(richText(item.body));
  if (item.youtubeId) node.append(video(item.youtubeId));
  if (item.canRemove) {
    node.append(el('button', {
      className: 'plain danger',
      type: 'button',
      onclick: async () => {
        if (!confirm('להסיר את התגובה? במקומה יישאר סימון שהוסרה.')) return;
        await api(`/comments/${item.id}`, { method: 'DELETE' });
        reload();
      },
    }, 'הסרה'));
  }
  return node;
}

async function postPage(id) {
  const { post, comments } = await api(`/posts/${id}`);
  const section = el('section', { className: 'card' },
    el('h2', {}, 'תגובות'),
    el('p', { className: 'muted' }, 'כל התגובות, לפי סדר הכתיבה.'),
    ...comments.map((item) => comment(item, route)));

  if (me?.name) {
    const text = el('textarea', { placeholder: 'כתיבת תגובה', maxLength: 2000, dir: 'auto' });
    const error = el('p', { className: 'error', hidden: true });
    section.append(el('form', {
      onsubmit: async (e) => {
        e.preventDefault();
        try {
          await api(`/posts/${id}/comments`, { method: 'POST', body: { body: text.value } });
          route();
        } catch (err) {
          error.textContent = err.message;
          error.hidden = false;
        }
      },
    }, text, el('div', { className: 'row' }, el('button', { type: 'submit' }, 'תגובה')), error));
  } else {
    section.append(el('p', {}, link('/login', 'התחברו'), ' כדי להגיב.'));
  }
  app.replaceChildren(postCard(post, { onDeleted: () => go('/') }), section);
}

function person(user, ...actions) {
  return el('div', { className: 'person' }, el('span', { dir: 'auto' }, user.name), el('div', { className: 'row' }, ...actions));
}

function friendAction(label, method, id, { plain = false } = {}) {
  return el('button', {
    className: plain ? 'plain' : '',
    type: 'button',
    onclick: async () => {
      await api(`/friends/${id}`, { method });
      route();
    },
  }, label);
}

async function friendsPage() {
  const { friends, incoming, outgoing } = await api('/friends');
  const known = new Set([...friends, ...incoming, ...outgoing].map((user) => user.id));
  const results = el('div');
  const search = el('input', { type: 'text', placeholder: 'חיפוש לפי שם', dir: 'auto' });
  search.addEventListener('input', async () => {
    const { users } = await api(`/users?q=${encodeURIComponent(search.value)}`);
    results.replaceChildren(...users.filter((user) => !known.has(user.id))
      .map((user) => person(user, friendAction('בקשת חברות', 'POST', user.id))));
  });

  const card = el('section', { className: 'card' }, el('h1', {}, 'חברים'), search, results);
  if (incoming.length) {
    card.append(el('h2', {}, 'בקשות אליך'), ...incoming.map((user) => person(user,
      friendAction('אישור', 'POST', user.id), friendAction('דחייה', 'DELETE', user.id, { plain: true }))));
  }
  if (outgoing.length) {
    card.append(el('h2', {}, 'ממתינות לאישור'), ...outgoing.map((user) => person(user,
      friendAction('ביטול', 'DELETE', user.id, { plain: true }))));
  }
  card.append(el('h2', {}, `החברים שלך (${friends.length})`), ...friends.map((user) => person(user,
    friendAction('הסרה', 'DELETE', user.id, { plain: true }))));
  app.replaceChildren(card);
}

function loginPage() {
  const email = el('input', { type: 'email', required: true, placeholder: 'כתובת מייל', dir: 'ltr' });
  const message = el('p', { className: 'muted' });
  app.replaceChildren(el('form', {
    className: 'card',
    onsubmit: async (e) => {
      e.preventDefault();
      try {
        const { devLink } = await api('/auth/request', { method: 'POST', body: { email: email.value } });
        message.replaceChildren(el('a', { href: devLink }, 'קישור כניסה (מצב פיתוח)'));
      } catch (err) {
        message.textContent = err.message;
      }
    },
  },
  el('h1', {}, 'כניסה'),
  el('p', { className: 'muted' }, 'הרשת החופשית של פרדס חנה-כרכור. בלי פרסומות, בלי סיסמה ובלי אלגוריתם נסתר.'),
  email,
  el('div', { className: 'row' }, el('button', { type: 'submit' }, 'שליחת קישור כניסה')),
  message));
}

function namePage() {
  const name = el('input', { type: 'text', required: true, maxLength: 60, placeholder: 'השם שיוצג לאחרים', dir: 'auto' });
  const error = el('p', { className: 'error', hidden: true });
  app.replaceChildren(el('form', {
    className: 'card',
    onsubmit: async (e) => {
      e.preventDefault();
      try {
        await api('/me', { method: 'PUT', body: { name: name.value } });
        await start();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
      }
    },
  }, el('h1', {}, 'ברוכים הבאים'), name, el('div', { className: 'row' }, el('button', { type: 'submit' }, 'המשך')), error));
}

function renderNav() {
  if (!me) {
    nav.replaceChildren(link('/login', 'כניסה'));
    return;
  }
  nav.replaceChildren(link('/friends', 'חברים'), el('button', {
    className: 'plain',
    type: 'button',
    onclick: async () => {
      await api('/auth/logout', { method: 'POST' });
      me = null;
      go('/login');
      renderNav();
    },
  }, 'יציאה'));
}

async function route() {
  const post = location.pathname.match(/^\/p\/(\d+)$/);
  try {
    if (post) await postPage(post[1]);
    else if (!me) loginPage();
    else if (!me.name) namePage();
    else if (location.pathname === '/friends') await friendsPage();
    else await feedPage();
  } catch (err) {
    app.replaceChildren(el('p', { className: 'error' }, err.message));
  }
}

async function start() {
  const token = new URLSearchParams(location.hash.slice(1)).get('token');
  if (token) {
    history.replaceState(null, '', '/');
    try {
      await api('/auth/verify', { method: 'POST', body: { token } });
    } catch (err) {
      alert(err.message);
    }
  }
  ({ user: me } = await api('/me'));
  renderNav();
  if (location.pathname === '/login' && me) history.replaceState(null, '', '/');
  await route();
}

window.addEventListener('popstate', route);
window.addEventListener('hashchange', start);
start();
