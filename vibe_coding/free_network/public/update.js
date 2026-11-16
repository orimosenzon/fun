// The update: what happened lately, in circles that widen from friends to the world.
// It is short on purpose and it ends. Nothing here is ranked: each list is newest first.

const BOARD_URL = 'https://pardespedia.info/wiki/עמוד_ראשי#אירועי_תרבות_ובילוי_קרובים';
const PLANS_URL = 'https://orimosenzon.github.io/fun/vibe_coding/transparency/ma-bonim/';
const HE_NEWS_URL = 'https://he.wikipedia.org/wiki/תבנית:חדשות_ואקטואליה';
const EN_NEWS_URL = 'https://en.wikipedia.org/wiki/Portal:Current_events';
const DAY_MS = 86400000;

const format = (options) => new Intl.DateTimeFormat('he-IL', options);

function localDay(ms) {
  const date = new Date(ms);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function eventDay(day) {
  if (day === localDay(Date.now())) return 'היום';
  if (day === localDay(Date.now() + DAY_MS)) return 'מחר';
  return format({ weekday: 'long' }).format(new Date(`${day}T12:00`));
}

function quietFor(last) {
  const months = Math.floor((Date.now() - last) / (30 * DAY_MS));
  if (months < 2) return 'יותר מחודש';
  return months === 2 ? 'חודשיים' : `${months} חודשים`;
}

export function updatePanel({ me, seen, ui }) {
  const { el, icon, avatar, link, api, when, tell, demo, flyGlobe } = ui;
  const body = el('div', { className: 'update-body' },
    el('div', { className: 'bone' }), el('div', { className: 'bone short' }), el('div', { className: 'bone' }));
  const root = el('section', { className: 'update' },
    el('header', { className: 'update-head' },
      el('h2', {}, 'העדכון'),
      el('span', { className: 'update-date' }, format({ weekday: 'long', day: 'numeric', month: 'long' }).format(Date.now()))),
    body);

  const outside = (url, ...content) => el('a', { className: 'update-item', href: url, target: '_blank', rel: 'noopener' }, ...content);
  const line = (main, small) => el('span', { className: 'update-text', dir: 'auto' }, main, small ? el('span', { className: 'update-small' }, small) : null);

  function personRow(user, main, small, path = `/u/${user.id}`) {
    const row = link(path, avatar(user, { small: true }), line(main, small));
    row.className = 'update-item';
    return row;
  }

  function ring(picture, title, ...children) {
    return el('section', { className: 'update-ring' }, el('h3', {}, icon(picture), title), ...children.flat().filter(Boolean));
  }

  // The reader's own line: write one, change it, or delete it.
  function statusBox() {
    const box = el('div', { className: 'update-status' });
    function draw(editing) {
      if (me.status && !editing) {
        box.replaceChildren(
          el('div', { className: 'update-item' }, avatar(me, { small: true }),
            line(me.status.text, el('span', {}, 'השורה שלך, ', when(me.status.at), '. נמחקת מעצמה אחרי שבוע.'))),
          el('div', { className: 'update-status-actions' },
            el('button', { className: 'plain', type: 'button', onclick: () => draw(true) }, 'שינוי'),
            el('button', {
              className: 'plain danger',
              type: 'button',
              onclick: async () => {
                try {
                  await api('/me/status', { method: 'DELETE' });
                  me.status = null;
                  draw(false);
                } catch (err) {
                  tell(err.message);
                }
              },
            }, 'מחיקה')));
        return;
      }
      const input = el('input', { type: 'text', maxLength: 140, dir: 'auto', placeholder: 'מה נשמע? משפט אחד לחברים', ariaLabel: 'מה נשמע? משפט אחד לחברים', value: me.status?.text ?? '' });
      box.replaceChildren(el('form', {
        onsubmit: async (e) => {
          e.preventDefault();
          if (!input.value.trim()) return;
          try {
            ({ status: me.status } = await api('/me/status', { method: 'PUT', body: { text: input.value } }));
            draw(false);
          } catch (err) {
            tell(err.message);
          }
        },
      }, input, el('button', { className: 'small', type: 'submit' }, 'שיתוף')),
      el('p', { className: 'update-source' }, 'רק החברים שלך רואים, והשורה נמחקת מעצמה אחרי שבוע.'));
      if (editing) input.focus();
    }
    draw(false);
    return box;
  }

  function friendsRing({ statuses, fresh, replies, moved, joined, quiet }) {
    const rows = [
      ...statuses.map((item) => personRow(item.user, el('strong', {}, item.user.name), el('span', { className: 'update-quote' }, item.text))),
      ...replies.map((item) => personRow(item.user, `תגובה מ${item.user.name} ${item.onMine ? 'לפוסט שלך' : 'לתגובה שלך'}`, when(item.at), `/p/${item.postId}`)),
      ...fresh.map((item) => personRow(item.user, item.user.name, item.count === 1 ? 'פוסט חדש שעוד לא ראית' : `${item.count} פוסטים חדשים שעוד לא ראית`)),
      ...moved.map((item) => {
        const row = el('button', { className: 'update-item', type: 'button', onclick: () => flyGlobe(item.place) },
          avatar(item.user, { small: true }), line(item.user.name, 'מקום חדש על הגלובוס. לחיצה מסובבת אותו לשם.'));
        return row;
      }),
      ...joined.map((item) => personRow(item.user, item.user.name, el('span', {}, 'חדש ברשת, מ', when(item.at)))),
      ...quiet.map((item) => personRow(item.user,
        item.last ? `לא שמעת מ${item.user.name} כבר ${quietFor(item.last)}` : `עוד לא שמעת כאן מ${item.user.name}`,
        'אולי זה הזמן לכתוב כמה מילים')),
    ];
    return ring('users', 'החברים שלי',
      demo ? null : statusBox(),
      rows.length ? rows : el('p', { className: 'update-none' }, 'שקט אצל החברים בימים האחרונים.'));
  }

  function communityRing({ events, moreEvents, plans }) {
    if (!events.length && !plans.length) return null;
    const short = (day) => format({ day: 'numeric', month: 'numeric' }).format(new Date(`${day}T12:00`));
    return ring('home', 'במושבה',
      events.length ? el('h4', {}, 'אירועים קרובים') : null,
      events.map((event) => {
        const content = line(event.name, [eventDay(event.day), event.time, event.venue].filter(Boolean).join(' · '));
        return event.url ? outside(event.url, content) : el('div', { className: 'update-item' }, content);
      }),
      events.length ? el('p', { className: 'update-source' },
        el('a', { href: BOARD_URL, target: '_blank', rel: 'noopener' }, moreEvents ? `עוד ${moreEvents} אירועים השבוע בפרדספדיה` : 'לוח האירועים בפרדספדיה')) : null,
      plans.length ? el('h4', {}, 'תוכניות בנייה שאפשר להתנגד להן') : null,
      plans.map((plan) => outside(plan.url, line(plan.title, [plan.adds, `עד ${short(plan.shuts)}`].filter(Boolean).join(' · ')))),
      plans.length ? el('p', { className: 'update-source' },
        el('a', { href: PLANS_URL, target: '_blank', rel: 'noopener' }, 'כל התוכניות ב"מה בונים לנו"')) : null);
  }

  function newsRing(title, items, sourceUrl, sourceLabel, dir) {
    if (!items.length) return null;
    return ring('globe', title,
      el('ul', { className: 'update-news', dir }, ...items.map((item) => el('li', {},
        item.url ? el('a', { href: item.url, target: '_blank', rel: 'noopener' }, item.text) : item.text))),
      el('p', { className: 'update-source' }, el('a', { href: sourceUrl, target: '_blank', rel: 'noopener' }, sourceLabel), ', ברישיון CC BY-SA'));
  }

  api(`/update${seen ? `?seen=${seen}` : ''}`).then((data) => {
    body.replaceChildren(...[
      friendsRing(data.friends),
      communityRing(data.community),
      newsRing('בארץ ובעולם', data.country, HE_NEWS_URL, 'מתוך ויקיפדיה העברית', 'rtl'),
      newsRing('בעולם, באנגלית', data.world, EN_NEWS_URL, 'מתוך ויקיפדיה האנגלית', 'ltr'),
      el('p', { className: 'update-end' }, 'זה כל העדכון. אין לו המשך לגלול אליו.')].filter(Boolean));
  }).catch((err) => {
    body.replaceChildren(el('p', { className: 'error' }, err.message));
  });
  return root;
}
