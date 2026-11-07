const app = document.getElementById('app');
const nav = document.getElementById('nav');
const MAX_IMAGE_SIDE = 1600;
const MAX_IMAGES = 4;
const MAX_REPORTS = 20;
let reports = 0;
const ICONS = {
  user: 'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 3a4 4 0 1 0 0 8a4 4 0 1 0 0-8',
  info: 'M12 22a10 10 0 1 0 0-20a10 10 0 0 0 0 20M12 16v-4M12 8h.01',
  logout: 'M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9',
  comment: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
  link: 'M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71',
  trash: 'M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6',
  image: 'M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM8.5 10a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3M21 15l-5-5L5 21',
  leaf: 'M11 20A7 7 0 0 1 4 13C4 7 10 4 20 4c0 10-3 16-9 16zM4 21c2-6 6-9 11-11',
};
let me = null;
let demo = false;

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

function link(path, ...content) {
  return el('a', { href: path, onclick: (e) => { e.preventDefault(); go(path); } }, ...content);
}

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', ICONS[name]);
  svg.append(path);
  return svg;
}

function labelled(name, text) {
  return el('span', { className: 'with-icon' }, icon(name), el('span', { className: 'label' }, text));
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
  const button = el('button', { className: 'video', type: 'button' }, el('span', { className: 'play' }), 'סרטון מיוטיוב, לחצו לטעינה');
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

function avatar(user, { small = false } = {}) {
  return el('span', { className: `avatar c${(user.id - 1) % 6}${small ? ' small' : ''}`, ariaHidden: 'true' }, [...user.name][0] ?? '?');
}

function when(ms) {
  return new Intl.DateTimeFormat('he-IL', { dateStyle: 'short', timeStyle: 'short' }).format(ms);
}

function postCard(post, { onDeleted } = {}) {
  const url = `/p/${post.id}`;
  const count = post.commentCount ? `תגובות (${post.commentCount})` : 'להגיב';
  const actions = el('div', { className: 'row spread' }, link(url, labelled('comment', count)));
  if (post.audience === 'public') {
    const copy = el('button', { className: 'plain', type: 'button' }, labelled('link', 'העתקת קישור'));
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
    }, labelled('trash', 'מחיקה')));
  }
  return el('article', { className: 'card' },
    el('div', { className: 'head' }, avatar(post.author), el('div', { className: 'who' },
      el('span', { className: 'author', dir: 'auto' }, post.author.name),
      el('span', { className: 'muted' }, `${when(post.createdAt)} · ${post.audience === 'public' ? 'ציבורי' : 'חברים'}`))),
    post.body ? richText(post.body) : null,
    post.images.length ? gallery(post.images) : null,
    post.youtubeId ? video(post.youtubeId) : null,
    actions);
}

async function composer(onPosted) {
  const { friends } = await api('/audience');
  const text = el('textarea', { placeholder: 'מה חדש?', maxLength: 5000, dir: 'auto' });
  const note = el('p', { className: 'note' });
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
  const addPhoto = el('button', { className: 'plain', type: 'button', onclick: () => picker.click() }, labelled('image', 'הוספת תמונה'));

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
  }, el('div', { className: 'composer-top' }, avatar(me), text), previews, picker, el('div', { className: 'row spread' }, choices, addPhoto, submit), note, error);
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
      list.append(el('div', { className: 'card empty' },
        el('h2', {}, 'שקט כאן בינתיים'),
        el('p', { className: 'muted' }, 'כתבו את הפוסט הראשון, או ', link('/friends', 'מצאו חברים'), ' מהשכונה.')));
    }
  }
  more.addEventListener('click', load);
  await load();

  app.replaceChildren(
    await composer(route),
    el('p', { className: 'feed-rule' }, 'כל הפוסטים שלך ושל החברים שלך, מהחדש לישן. זה כל האלגוריתם.'),
    list,
    more);
}

function comment(item, reload) {
  const bubble = el('div', { className: 'bubble' },
    el('div', { className: 'comment-head' },
      el('span', { className: 'author', dir: 'auto' }, item.author.name),
      el('span', { className: 'muted' }, when(item.createdAt))));
  const node = el('div', { className: 'comment' }, avatar(item.author, { small: true }), bubble);
  if (item.removedBy) {
    const by = item.removedBy === 'author' ? 'הכותב' : 'בעל הפוסט';
    bubble.append(el('p', { className: 'text removed' }, `התגובה הוסרה על ידי ${by}`));
    return node;
  }
  bubble.append(richText(item.body));
  if (item.youtubeId) bubble.append(video(item.youtubeId));
  if (item.canRemove) {
    bubble.append(el('button', {
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
  const section = el('section', { className: 'card thread' },
    el('h2', {}, 'תגובות', icon('leaf')),
    el('p', { className: 'muted' }, 'כל התגובות, לפי סדר הכתיבה.'),
    ...comments.map((item) => comment(item, route)));

  if (me?.name) {
    const text = el('textarea', { placeholder: 'כתיבת תגובה', maxLength: 2000, dir: 'auto' });
    const error = el('p', { className: 'error', hidden: true });
    section.append(el('form', {
      style: 'margin-top: 16px',
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
  return el('div', { className: 'person' },
    el('div', { className: 'head' }, avatar(user, { small: true }), el('span', { dir: 'auto' }, user.name)),
    el('div', { className: 'row', style: 'margin-top: 0' }, ...actions));
}

function friendAction(label, method, id, { plain = false } = {}) {
  return el('button', {
    className: plain ? 'plain' : '',
    type: 'button',
    onclick: async () => {
      try {
        await api(`/friends/${id}`, { method });
        route();
      } catch (err) {
        alert(err.message);
      }
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
  const demoEntry = el('div', { className: 'card' },
    el('p', { style: 'margin-top: 0' }, 'כאן אפשר להציץ בממשק עם אנשים ופוסטים בדויים.'),
    el('div', { className: 'row' }, el('button', {
      type: 'button',
      onclick: async () => {
        await api('/auth/demo', { method: 'POST' });
        await start();
        go('/');
      },
    }, 'כניסה לדמו')));
  app.replaceChildren(el('section', { className: 'hero' },
    el('img', { src: '/favicon.svg', alt: '' }),
    el('h1', {}, 'השכונה שלך, בלי הרעש'),
    el('p', {}, 'הרשת החופשית של פרדס חנה-כרכור. מקום לחברים ולשכנים, שלא מנסה למכור לך כלום.')),
  demo ? demoEntry : el('form', {
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
  el('p', { style: 'margin-top: 0' }, 'נכנסים עם קישור למייל, בלי סיסמה.'),
  email,
  el('div', { className: 'row' }, el('button', { type: 'submit' }, 'שליחת קישור כניסה')),
  message),
  el('ul', { className: 'promises' },
    el('li', {}, el('strong', {}, 'החברים שלך באמת יראו'), 'כל פוסט מגיע לפיד של כל החברים, בלי סינון.'),
    el('li', {}, el('strong', {}, 'כל התגובות, לפי הסדר'), 'בלי "הרלוונטיות ביותר" ובלי תגובות שנעלמות.'),
    el('li', {}, el('strong', {}, 'בלי פרסומות'), 'עמותה ללא מטרות רווח. אין מי שמרוויח מהזמן שלך.')),
  demo ? '' : el('p', { className: 'feed-rule' }, el('a', { href: '/why' }, 'למה בכלל צריך רשת חדשה?')));
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
  const why = demo ? null : el('a', { href: '/why' }, labelled('info', 'למה אנחנו'));
  if (!me) {
    nav.replaceChildren(...[why, link('/login', 'כניסה')].filter(Boolean));
    return;
  }
  nav.replaceChildren(...[link('/friends', labelled('user', 'חברים')), why, el('button', {
    className: 'plain',
    type: 'button',
    onclick: async () => {
      await api('/auth/logout', { method: 'POST' });
      me = null;
      go('/login');
      renderNav();
    },
  }, labelled('logout', 'יציאה'))].filter(Boolean));
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
  ({ user: me, demo } = await api('/me'));
  document.getElementById('demo-banner').hidden = !demo;
  renderNav();
  if (location.pathname === '/login' && me) history.replaceState(null, '', '/');
  await route();
}

window.addEventListener('popstate', route);
window.addEventListener('hashchange', start);
start();
