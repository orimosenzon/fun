// Notices: everything that others did around the member, one line for each thing, the newest
// first and back to the very first. Nothing is folded into "and 5 others" and nothing is chosen
// for the reader. The same page is where a device asks to be told while the network is closed.

import { face, feelingLabel } from '/reactions.js';

const GROUPS = [['', 'הכול'], ['comments', 'תגובות'], ['feelings', 'רגשות'], ['shares', 'שיתופים'], ['mentions', 'תיוגים'], ['friends', 'חברויות']];
const PUSHED = [['messages', 'הודעות פרטיות'], ['comments', 'תגובות ותשובות'], ['feelings', 'רגשות'], ['shares', 'שיתופים'], ['mentions', 'תיוגים'], ['friends', 'חברויות']];
const BADGE = { comment: 'comment', reply: 'reply', thread: 'comment', share: 'repost', reshare: 'repost', request: 'users', accepted: 'check', joined: 'plus', mention: 'at' };
const HINT_KEY = 'push-hint-hidden';

// ---------- Pushes to this device ----------

const supported = () => 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window;
const keyBytes = (key) => Uint8Array.from(atob(key.replace(/-/g, '+').replace(/_/g, '/')), (ch) => ch.charCodeAt(0));

let worker = null;
function pushWorker() {
  worker ??= navigator.serviceWorker.register('/sw.js');
  return worker;
}

// The subscription of this device, if it was made with the key the network signs with today.
async function subscription(key) {
  const found = await (await pushWorker()).pushManager.getSubscription();
  if (!found) return null;
  const used = new Uint8Array(found.options.applicationServerKey ?? []);
  const wanted = keyBytes(key);
  if (used.length === wanted.length && used.every((byte, i) => byte === wanted[i])) return found;
  await found.unsubscribe();
  return null;
}

// 'none': this browser cannot be pushed to. 'blocked': the person refused. 'off' and 'on'.
export async function pushState(key) {
  if (!key || !supported()) return 'none';
  if (Notification.permission === 'denied') return 'blocked';
  if (Notification.permission !== 'granted') return 'off';
  return (await subscription(key)) ? 'on' : 'off';
}

async function pushOn(key, api) {
  const permission = await Notification.requestPermission();
  if (permission !== 'granted') return permission === 'denied' ? 'blocked' : 'off';
  const registration = await pushWorker();
  await navigator.serviceWorker.ready;
  const made = (await subscription(key)) ?? await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(key) });
  await api('/push', { method: 'PUT', body: made.toJSON() });
  return 'on';
}

async function pushOff(api) {
  const found = await (await pushWorker()).pushManager.getSubscription();
  if (!found) return;
  await api('/push', { method: 'DELETE', body: { endpoint: found.endpoint } });
  await found.unsubscribe();
}

// A device that asked before is tied to whoever is logged in on it now. Logging out cut the
// tie on the server, and this is what restores it, with no question asked twice.
export async function pushKeep(key, api) {
  if (!key || !supported() || Notification.permission !== 'granted') return;
  const found = await subscription(key);
  if (found) await api('/push', { method: 'PUT', body: found.toJSON() });
}

// What the service worker tells an open page: a push arrived, or a notification was pressed.
export function pushListen({ onPush, onOpen }) {
  if (!supported()) return;
  navigator.serviceWorker.addEventListener('message', (event) => {
    if (event.data?.type === 'push') onPush(event.data.message ?? {}, !!event.data.shown);
    if (event.data?.type === 'open') onOpen(event.data.path);
  });
}

function hintHidden(value) {
  try {
    if (value) localStorage.setItem(HINT_KEY, '1');
    return !!localStorage.getItem(HINT_KEY);
  } catch {
    return true;
  }
}

// ---------- The card that turns pushes on and off ----------

function pushCard({ me, pushKey, ui, compact = false }) {
  const { el, icon, link, api, tell, toast } = ui;
  const box = el('section', { className: `card push-card${compact ? ' compact' : ''}` });
  const say = (title, ...rest) => box.replaceChildren(el('span', { className: 'promise-icon' }, icon('bell')),
    el('div', { className: 'push-words' }, el('strong', {}, title), ...rest));
  const failed = (err) => tell(`לא הצלחנו. ${err.message}`);

  async function turnOn(button) {
    button.disabled = true;
    try {
      draw(await pushOn(pushKey, api));
    } catch (err) {
      ui.report(`push subscribe: ${err.message}`);
      tell('הדפדפן הזה לא הצליח להירשם להתראות. אפשר לנסות שוב, או בדפדפן אחר.');
      button.disabled = false;
    }
  }

  function kinds() {
    const boxes = PUSHED.map(([group, label]) => {
      const input = el('input', { type: 'checkbox', checked: !me.pushMuted.includes(group) });
      input.addEventListener('change', async () => {
        const muted = PUSHED.map(([name], i) => (boxes[i].input.checked ? null : name)).filter(Boolean);
        try {
          ({ muted: me.pushMuted } = await api('/push/muted', { method: 'PUT', body: { muted } }));
        } catch (err) {
          input.checked = !input.checked;
          failed(err);
        }
      });
      return { input, node: el('label', { className: 'push-kind' }, input, label) };
    });
    return el('details', { className: 'push-kinds' },
      el('summary', {}, 'על מה להתריע'),
      ...boxes.map((item) => item.node),
      el('p', { className: 'note' }, 'הבחירה חלה על כל המכשירים שלך. בדף הזה הכול מופיע תמיד, גם מה שלא מתריעים עליו.'));
  }

  function draw(state) {
    box.hidden = false;
    if (state === 'on') {
      if (compact) {
        box.remove();
        toast('ההתראות פועלות במכשיר הזה.', link('/notices', 'הגדרות'));
        return;
      }
      say('ההתראות פועלות במכשיר הזה',
        el('span', { className: 'muted' }, 'תגיע התראה גם כשהרשת סגורה. בכל מכשיר מפעילים בנפרד.'));
      box.append(el('div', { className: 'push-actions' },
        el('button', {
          className: 'small',
          type: 'button',
          onclick: async (e) => {
            const button = e.currentTarget;
            button.disabled = true;
            try {
              const { delivered } = await api('/push/test', { method: 'POST' });
              if (delivered) toast('נשלחה. היא אמורה להופיע בתוך כמה שניות.');
              else tell('ההתראה לא יצאה: המכשיר הזה כבר לא רשום. כבו והפעילו שוב את ההתראות.');
            } catch (err) {
              failed(err);
            }
            button.disabled = false;
          },
        }, 'התראה לדוגמה'),
        el('button', {
          className: 'plain',
          type: 'button',
          onclick: async () => {
            try {
              await pushOff(api);
              draw('off');
            } catch (err) {
              failed(err);
            }
          },
        }, 'כיבוי במכשיר הזה')), kinds());
    } else if (state === 'off') {
      const button = el('button', { className: 'small', type: 'button', onclick: () => turnOn(button) }, 'הפעלת התראות');
      say(compact ? 'לדעת כשכותבים לך' : 'לדעת גם כשהרשת סגורה',
        el('span', { className: 'muted' }, 'התראה במכשיר הזה כשמגיבים לך, משתפים פוסט שלך או כותבים לך בפרטי.'));
      box.append(el('div', { className: 'push-actions' }, button, compact ? el('button', {
        className: 'plain',
        type: 'button',
        onclick: () => {
          hintHidden(true);
          box.remove();
        },
      }, 'לא עכשיו') : null));
    } else if (compact) {
      box.remove();
    } else if (state === 'blocked') {
      say('ההתראות חסומות במכשיר הזה', ui.phone === 'android' && ui.installed
        ? el('span', { className: 'muted' }, 'מאפשרים אותן בהגדרות הטלפון, תחת האפליקציות, ב"הרשת החופשית". אם האפליקציה הותקנה לפני שנוספו לה התראות, צריך קודם ',
          el('a', { href: ui.apk, download: 'free-network.apk' }, 'להתקין את הגרסה החדשה'), '.')
        : el('span', { className: 'muted' }, 'כדי להפעיל אותן מאפשרים לאתר להציג התראות, בהגדרות האתר שבדפדפן (המנעול שליד הכתובת), ואז טוענים את הדף מחדש.'));
    } else if (ui.phone === 'ios' && !ui.installed) {
      say('התראות באייפון', el('span', { className: 'muted' }, 'אייפון מאפשר התראות רק כשהרשת נמצאת במסך הבית. ', link('/app', 'איך מוסיפים אותה?')));
    } else if (pushKey) {
      say('בלי התראות בדפדפן הזה', el('span', { className: 'muted' }, 'הדפדפן הזה לא יודע להתריע כשהרשת סגורה. כל מה שקורה מחכה כאן, בדף הזה.'));
    } else {
      box.remove();
    }
  }

  box.hidden = true;
  pushState(pushKey).then(draw, () => box.remove());
  return box;
}

// A small card above the feed, once, for a device that can be pushed to and was not asked yet.
export function pushHint(options) {
  if (!options.pushKey || !supported() || Notification.permission !== 'default' || hintHidden()) return null;
  return pushCard({ ...options, compact: true });
}

// ---------- The page ----------

export function noticesPage({ me, pushKey, ui }) {
  const { el, icon, avatar, link, api, when, dayLabel, seen } = ui;
  const list = el('div', { className: 'notices' });
  const more = el('button', { className: 'ghost more', type: 'button', hidden: true }, 'התראות ישנות יותר');
  const end = el('p', { className: 'feed-rule', hidden: true }, 'אלה כל ההתראות, מהראשונה.');
  let group = '';
  let before = null;
  let lastDay = null;
  let turn = 0;

  function row(item) {
    const badge = item.feeling
      ? el('span', { className: 'notice-badge feeling', title: feelingLabel(item.feeling) }, face(item.feeling))
      : el('span', { className: `notice-badge ${item.kind}` }, icon(BADGE[item.kind]));
    const node = link(item.path,
      el('span', { className: 'notice-face' }, avatar(item.actor), badge),
      el('span', { className: 'notice-body' },
        el('span', { className: 'notice-line' },
          el('strong', { dir: 'auto' }, item.actor.name), ' ',
          item.feeling ? el('span', { className: 'notice-feeling' }, face(item.feeling)) : null, item.text),
        item.words || item.image ? el('span', { className: 'notice-words', dir: 'auto' }, item.words ?? 'תמונה') : null,
        item.about ? el('span', { className: 'notice-about', dir: 'auto' }, item.about) : null,
        el('span', { className: 'notice-when' }, when(item.at))),
      item.fresh ? el('span', { className: 'notice-dot', title: 'חדש' }) : null);
    node.className = `notice${item.fresh ? ' fresh' : ''}`;
    return node;
  }

  // The first page of a filter replaces the list, and the pages after it join at its end.
  async function load(first) {
    const mine = first ? ++turn : turn;
    const query = new URLSearchParams({ ...(group ? { group } : {}), ...(first || !before ? {} : { before }) });
    const data = await api(`/notices?${query}`);
    if (mine !== turn) return;
    if (first) {
      list.replaceChildren();
      lastDay = null;
      // Whatever the filter, the member has now been shown that something new came.
      seen(data.now);
    }
    for (const item of data.notices) {
      const day = dayLabel(item.at);
      if (day !== lastDay) list.append(el('h2', { className: 'day' }, el('span', {}, day)));
      lastDay = day;
      list.append(row(item));
    }
    before = data.before;
    more.hidden = !before;
    end.hidden = !!before || !list.children.length;
    if (!list.children.length) {
      list.append(el('div', { className: 'empty notices-empty' },
        el('img', { src: '/favicon.svg', alt: '' }),
        el('p', { className: 'muted' }, group ? 'אין התראות מהסוג הזה.' : 'עוד אין התראות. כשמישהו יגיב לך, ישתף פוסט שלך או יבקש חברות, זה יופיע כאן.')));
    }
  }

  const chips = el('div', { className: 'notice-filter', role: 'group', ariaLabel: 'סינון ההתראות' }, ...GROUPS.map(([name, label]) => el('button', {
    className: `plain${name === group ? ' chosen' : ''}`,
    type: 'button',
    onclick: async (e) => {
      group = name;
      for (const chip of chips.children) chip.classList.toggle('chosen', chip === e.currentTarget);
      await load(true).catch((err) => ui.tell(err.message));
    },
  }, label)));

  more.addEventListener('click', async () => {
    more.disabled = true;
    await load(false).catch((err) => ui.tell(err.message));
    more.disabled = false;
  });

  // Something new came while the page is open: it joins the top by itself.
  const fresh = () => load(true).catch(() => {});
  document.addEventListener('notices', fresh);

  return {
    ready: load(true),
    node: el('div', { className: 'notices-page' },
      pushCard({ me, pushKey, ui }),
      el('section', { className: 'card' },
        el('h1', {}, 'התראות'),
        el('p', { className: 'muted' }, 'כל מה שקרה סביבך, שורה לכל דבר, מהחדש לישן.'),
        chips, list, more, end)),
    leave: () => document.removeEventListener('notices', fresh),
  };
}
