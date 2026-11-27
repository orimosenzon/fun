// Private messages: the list of conversations, and one of them open beside it. On a narrow
// screen the two are separate pages. There is no live connection: an open conversation asks
// the server for news every few seconds, slows down when nothing happens, and stops asking
// while the page is hidden.

import { keepWhole } from '/limit.js';

const FAST_MS = 3000;
const SLOW_MS = 12000;
const CALM_AFTER_MS = 2 * 60 * 1000;
// Messages from the same side within a few minutes sit close together.
const NEAR_MS = 5 * 60 * 1000;
const BOTTOM_SLACK = 60;

const format = (options) => new Intl.DateTimeFormat('he-IL', options);
const hour = (ms) => format({ timeStyle: 'short' }).format(ms);

export async function messenger({ me, withId, ui }) {
  const { el, icon, avatar, link, api, ask, tell, richText, lightbox, prepareImage, dayLabel, daysAgo, back, pulse, unread } = ui;
  const [{ conversations }, open] = await Promise.all([api('/messages'), withId ? api(`/messages/${withId}`) : null]);
  let convos = conversations;

  function shortTime(ms) {
    const days = daysAgo(ms);
    if (days === 0) return hour(ms);
    if (days === 1) return 'אתמול';
    if (days < 7) return format({ weekday: 'long' }).format(ms);
    return format({ day: 'numeric', month: 'numeric', year: '2-digit' }).format(ms);
  }

  function snippet(last) {
    if (last.removed) return 'ההודעה נמחקה';
    const text = last.body ? last.body.replace(/\s+/g, ' ') : 'תמונה';
    return `${last.mine ? 'אני: ' : ''}${text}`;
  }

  const rows = el('div', { className: 'convo-rows' });
  const none = el('p', { className: 'note', hidden: true }, 'עוד אין לך שיחות.');
  function showRows() {
    rows.replaceChildren(...convos.map((item) => {
      const row = link(`/messages/${item.user.id}`,
        avatar(item.user),
        el('span', { className: 'convo-text' },
          el('span', { className: 'convo-name', dir: 'auto' }, item.user.name),
          el('span', { className: 'convo-last', dir: 'auto' }, snippet(item.last))),
        el('span', { className: 'convo-side' },
          el('span', { className: 'convo-time' }, shortTime(item.last.at)),
          item.unread ? el('span', { className: 'count', ariaLabel: `${item.unread} הודעות שלא נקראו` }, String(item.unread)) : null));
      row.className = `convo${item.user.id === withId ? ' active' : ''}${item.unread ? ' unread' : ''}`;
      if (item.user.id === withId) row.ariaCurrent = 'page';
      return row;
    }));
    none.hidden = convos.length > 0;
  }

  // Starting a conversation: the reader's friends, narrowed by typing a name.
  const picker = el('div', { className: 'convo-picker', hidden: true });
  let friends = null;
  async function openPicker(focus) {
    try {
      friends ??= (await api('/friends')).friends;
    } catch (err) {
      tell(err.message);
      return;
    }
    const found = el('div');
    const filter = el('input', { type: 'search', placeholder: 'למי לכתוב?', dir: 'auto', ariaLabel: 'חיפוש חבר לפי שם' });
    const show = () => {
      const q = filter.value.trim().toLowerCase();
      const list = friends.filter((friend) => friend.name.toLowerCase().includes(q));
      found.replaceChildren(...list.map((friend) => {
        const row = link(`/messages/${friend.id}`, avatar(friend, { small: true }), el('span', { className: 'convo-name', dir: 'auto' }, friend.name));
        row.className = 'convo';
        return row;
      }));
      if (!list.length) found.append(el('p', { className: 'note' }, 'אין חבר בשם הזה.'));
    };
    filter.addEventListener('input', show);
    show();
    picker.replaceChildren(...(friends.length
      ? [el('label', { className: 'search' }, icon('search'), filter), found]
      : [el('p', { className: 'note' }, 'הודעות כותבים לחברים, ועוד אין לך חברים כאן. ', link('/friends', 'לחיפוש חברים'))]));
    picker.hidden = false;
    if (focus) filter.focus();
  }

  const start = el('button', {
    className: 'small',
    type: 'button',
    ariaExpanded: 'false',
    onclick: async () => {
      if (picker.hidden) await openPicker(true);
      else picker.hidden = true;
      start.ariaExpanded = String(!picker.hidden);
    },
  }, el('span', { className: 'with-icon' }, icon('plus'), 'שיחה חדשה'));

  const listPanel = el('section', { className: 'card convo-list' },
    el('div', { className: 'convo-head' }, el('h1', {}, 'הודעות'), start),
    picker, none, rows,
    el('p', { className: 'convo-rule' }, 'שיחה רואים רק שני הצדדים שלה. ההודעות נשמרות בשרת של הרשת, בלי הצפנה מקצה לקצה.'));
  showRows();
  if (!convos.length && !open) openPicker(false).then(() => { start.ariaExpanded = String(!picker.hidden); });

  const chat = open ? chatPanel(open) : null;
  const node = el('div', { className: `messenger${open ? ' has-chat' : ''}` },
    listPanel,
    chat?.node ?? el('section', { className: 'card chat chat-none' },
      el('img', { src: '/favicon.svg', alt: '' }),
      el('p', {}, 'בחרו שיחה מהרשימה, או התחילו שיחה חדשה.')));

  // The count on the menu changed, so something was written in another conversation.
  const reload = async () => {
    try {
      convos = (await api('/messages')).conversations;
      showRows();
    } catch {
      // The list stays as it was until the next change.
    }
  };
  document.addEventListener('unread', reload);

  function chatPanel({ user: other, messages, hasMore: more, canSend, readUpTo, now }) {
    let list = [...messages];
    let hasMore = more;
    let since = now;
    let newest = list.at(-1)?.id ?? 0;
    let markedUpTo = readUpTo;
    let pinned = true;
    let freshBelow = false;
    let stopped = false;
    let busy = false;
    let timer = null;
    let lastActivity = Date.now();
    let queue = Promise.resolve();
    let nextKey = 0;
    const pending = [];
    const nodes = new Map();
    const firstUnread = list.find((item) => !item.mine && !item.removed && item.id > readUpTo)?.id ?? null;
    const coarse = matchMedia('(pointer: coarse)').matches;
    const draftKey = `draft:${me.id}:${other.id}`;

    const log = el('div', { className: 'chat-log', role: 'group', ariaLabel: `השיחה עם ${other.name}` });
    const toBottom = () => { log.scrollTop = log.scrollHeight; };
    const unreadMark = el('p', { className: 'seen-mark chat-unread' }, el('span', {}, 'הודעות חדשות'));
    const empty = el('p', { className: 'chat-empty' }, canSend ? `זו תחילת השיחה עם ${other.name}.` : 'אין כאן הודעות.');
    const jump = el('button', {
      className: 'chat-jump',
      type: 'button',
      onclick: () => {
        pinned = true;
        freshBelow = false;
        render();
      },
    }, 'הודעה חדשה', icon('down'));
    log.addEventListener('scroll', () => {
      pinned = log.scrollHeight - log.scrollTop - log.clientHeight < BOTTOM_SLACK;
      if (pinned && freshBelow) {
        freshBelow = false;
        jump.remove();
      }
    });

    const older = el('button', {
      className: 'plain chat-older',
      type: 'button',
      onclick: async () => {
        older.disabled = true;
        try {
          const page = await api(`/messages/${other.id}?before=${list[0].id}`);
          const fromBottom = log.scrollHeight - log.scrollTop;
          list = [...page.messages, ...list];
          hasMore = page.hasMore;
          pinned = false;
          render();
          log.scrollTop = log.scrollHeight - fromBottom;
        } catch (err) {
          tell(err.message);
        }
        older.disabled = false;
      },
    }, 'הודעות ישנות יותר');

    function picture(item) {
      const image = el('img', { src: item.preview ?? `/api/images/${item.image}`, alt: 'תמונה', onload: () => { if (pinned) toBottom(); } });
      if (!item.image) return el('span', { className: 'msg-picture' }, image);
      return el('a', {
        className: 'msg-picture',
        href: `/api/images/${item.image}`,
        target: '_blank',
        onclick: (e) => {
          if (e.ctrlKey || e.metaKey || e.shiftKey) return;
          e.preventDefault();
          lightbox([item.image], 0);
        },
      }, image);
    }

    function build(item) {
      const side = item.mine ? 'mine' : 'theirs';
      const time = el('time', {
        className: 'msg-time',
        dateTime: new Date(item.at).toISOString(),
        title: format({ dateStyle: 'full', timeStyle: 'short' }).format(item.at),
      }, item.pending && !item.failed ? 'בשליחה…' : hour(item.at));
      if (item.removed) {
        return el('div', { className: `msg-row ${side}` },
          el('div', { className: 'msg removed' }, el('p', { className: 'text' }, item.mine ? 'מחקת את ההודעה' : 'ההודעה נמחקה'), time));
      }
      const removable = item.mine && !item.pending;
      const box = el('div', { className: 'msg', ...(removable ? { tabIndex: 0 } : {}) });
      if (item.image || item.preview) box.append(picture(item));
      if (item.body) box.append(richText(item.body));
      box.append(time);
      const row = el('div', { className: `msg-row ${side}${item.pending ? ' pending' : ''}${item.failed ? ' failed' : ''}` }, box);
      if (removable) {
        row.append(el('button', {
          className: 'msg-remove',
          type: 'button',
          ariaLabel: 'מחיקת ההודעה',
          title: 'מחיקת ההודעה',
          onclick: async () => {
            if (!await ask(`למחוק את ההודעה? היא תימחק גם אצל ${other.name}, ובמקומה יישאר סימון שנמחקה.`, 'מחיקה')) return;
            try {
              await api(`/messages/${other.id}/${item.id}`, { method: 'DELETE' });
              erase(item.id);
              render();
              touchList();
            } catch (err) {
              tell(err.message);
            }
          },
        }, icon('trash')));
      }
      if (item.failed) {
        row.append(el('div', { className: 'msg-failed' },
          el('span', { className: 'error' }, `לא נשלח. ${item.error}`),
          el('button', {
            className: 'plain',
            type: 'button',
            onclick: () => {
              item.failed = false;
              nodes.delete(item.key);
              render();
              queue = queue.then(() => send(item));
            },
          }, 'לנסות שוב'),
          el('button', {
            className: 'plain danger',
            type: 'button',
            onclick: () => {
              pending.splice(pending.indexOf(item), 1);
              render();
            },
          }, 'ביטול')));
      }
      return row;
    }

    // Every message keeps its own element between redraws, so a picture is not loaded twice
    // and a redraw only moves things.
    function render() {
      const children = hasMore ? [older] : [];
      let day = null;
      let previous = null;
      for (const item of [...list, ...pending]) {
        const label = dayLabel(item.at);
        if (label !== day) {
          children.push(el('p', { className: 'chat-day' }, el('span', {}, label)));
          previous = null;
        }
        day = label;
        if (item.id === firstUnread) {
          children.push(unreadMark);
          previous = null;
        }
        const key = item.id ?? item.key;
        if (!nodes.has(key)) nodes.set(key, build(item));
        const row = nodes.get(key);
        row.classList.toggle('follows', !!previous && previous.mine === item.mine && item.at - previous.at < NEAR_MS);
        children.push(row);
        previous = item;
      }
      if (!list.length && !pending.length) children.push(empty);
      if (freshBelow) children.push(jump);
      log.replaceChildren(...children);
      if (pinned) toBottom();
    }

    function erase(id) {
      const item = list.find((known) => known.id === id);
      if (!item || item.removed) return false;
      Object.assign(item, { removed: true, body: null, image: null, preview: null });
      nodes.delete(id);
      return true;
    }

    // The row of this conversation in the list follows what happens here, without asking the server.
    function touchList() {
      const last = list.at(-1);
      if (!last) return;
      const seenHere = !document.hidden;
      const before = convos.find((item) => item.user.id === other.id);
      convos = [
        { user: other, last: { mine: last.mine, body: last.body, image: !!last.image, at: last.at, removed: last.removed }, unread: seenHere ? 0 : before?.unread ?? 0 },
        ...convos.filter((item) => item !== before),
      ].sort((a, b) => b.last.at - a.last.at);
      showRows();
    }

    // Tells the server how far the reader has seen, so the count is right on every device.
    async function seen() {
      if (stopped || document.hidden) return;
      const incoming = list.findLast((item) => !item.mine && !item.removed)?.id ?? 0;
      if (incoming <= markedUpTo) return;
      const before = markedUpTo;
      markedUpTo = list.at(-1).id;
      try {
        await api(`/messages/${other.id}/read`, { method: 'PUT', body: { upTo: markedUpTo } });
        touchList();
        if (unread()) pulse();
      } catch {
        markedUpTo = before;
      }
    }

    function apply(data) {
      since = data.now;
      if (data.canSend !== canSend) {
        canSend = data.canSend;
        showForm();
      }
      let changed = false;
      for (const item of data.messages) {
        newest = Math.max(newest, item.id);
        if (list.some((known) => known.id === item.id)) continue;
        list.push(item);
        changed = true;
      }
      for (const id of data.removed) changed = erase(id) || changed;
      if (!changed) return;
      list.sort((a, b) => a.id - b.id);
      if (data.messages.some((item) => !item.mine)) {
        lastActivity = Date.now();
        if (!pinned) freshBelow = true;
      }
      render();
      touchList();
      seen();
    }

    function schedule(delay) {
      clearTimeout(timer);
      if (stopped || document.hidden) return;
      timer = setTimeout(poll, delay ?? (Date.now() - lastActivity > CALM_AFTER_MS ? SLOW_MS : FAST_MS));
    }

    async function poll() {
      if (busy) return;
      busy = true;
      let wait;
      try {
        const data = await api(`/messages/${other.id}?after=${newest}&since=${since}`);
        if (!stopped) apply(data);
      } catch {
        // No connection right now. Asking again can wait a little longer.
        wait = SLOW_MS;
      }
      busy = false;
      schedule(wait);
    }

    async function send(item) {
      try {
        if (item.blob && !item.image) item.image = (await api('/images', { method: 'POST', body: item.blob })).id;
        const { message } = await api(`/messages/${other.id}`, { method: 'POST', body: { body: item.body ?? '', image: item.image ?? null } });
        pending.splice(pending.indexOf(item), 1);
        nodes.delete(item.key);
        // The regular asking may have brought the message already.
        if (!list.some((known) => known.id === message.id)) {
          list.push({ ...message, preview: item.preview });
          list.sort((a, b) => a.id - b.id);
        }
        markedUpTo = Math.max(markedUpTo, message.id);
        touchList();
      } catch (err) {
        item.failed = true;
        item.error = err.message;
        nodes.delete(item.key);
      }
      if (!stopped) render();
    }

    const text = el('textarea', {
      placeholder: 'כתיבת הודעה',
      rows: 1,
      dir: 'auto',
      ariaLabel: `הודעה אל ${other.name}`,
      enterKeyHint: coarse || !ui.prefs().enterMessages ? 'enter' : 'send',
    });
    keepWhole(text);
    const grow = () => {
      if (CSS.supports('field-sizing', 'content')) return;
      text.style.height = 'auto';
      text.style.height = `${text.scrollHeight + 3}px`;
    };
    const draft = (value) => {
      try {
        if (value === undefined) return sessionStorage.getItem(draftKey) ?? '';
        if (value) sessionStorage.setItem(draftKey, value);
        else sessionStorage.removeItem(draftKey);
      } catch {
        // Without storage, what was typed is simply not kept between pages.
      }
      return '';
    };
    text.value = draft();

    let photo = null;
    const error = el('p', { className: 'error', hidden: true });
    const previews = el('div', { className: 'previews' });
    const showPhoto = () => previews.replaceChildren(...(photo ? [el('button', {
      className: 'preview',
      type: 'button',
      title: 'הסרת התמונה',
      onclick: () => {
        photo = null;
        showPhoto();
      },
    }, el('img', { src: URL.createObjectURL(photo), alt: 'תמונה שנבחרה' }))] : []));
    async function attach(file) {
      error.hidden = true;
      try {
        photo = await prepareImage(file);
        showPhoto();
        text.focus();
      } catch {
        error.textContent = 'אי אפשר לקרוא את התמונה הזו';
        error.hidden = false;
      }
    }
    const chooser = el('input', { type: 'file', accept: 'image/*', hidden: true });
    chooser.addEventListener('change', async () => {
      if (chooser.files[0]) await attach(chooser.files[0]);
      chooser.value = '';
    });

    function submit() {
      if (!text.reportValidity()) return;
      const body = text.value.trim();
      if (!body && !photo) return;
      const item = { key: `pending-${nextKey++}`, mine: true, pending: true, body: body || null, blob: photo, preview: photo ? URL.createObjectURL(photo) : null, at: Date.now() };
      pending.push(item);
      text.value = '';
      draft('');
      grow();
      photo = null;
      showPhoto();
      error.hidden = true;
      pinned = true;
      freshBelow = false;
      lastActivity = Date.now();
      render();
      queue = queue.then(() => send(item));
      text.focus();
      schedule();
    }

    text.addEventListener('input', () => {
      lastActivity = Date.now();
      draft(text.value);
      grow();
    });
    // On a keyboard Enter sends and Shift+Enter breaks the line, unless the member chose in their
    // preferences that Enter breaks the line, and then Ctrl+Enter sends. On a phone Enter breaks
    // the line, and the button sends.
    text.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' || e.isComposing) return;
      if (!e.ctrlKey && !e.metaKey && (e.shiftKey || coarse || !ui.prefs().enterMessages)) return;
      e.preventDefault();
      submit();
    });
    text.addEventListener('paste', (e) => {
      const file = [...(e.clipboardData?.files ?? [])].find((item) => item.type.startsWith('image/'));
      if (!file) return;
      e.preventDefault();
      attach(file);
    });

    const form = el('form', {
      className: 'chat-form',
      onsubmit: (e) => {
        e.preventDefault();
        submit();
      },
    }, previews, chooser,
    el('button', { className: 'chat-attach', type: 'button', ariaLabel: 'צירוף תמונה', title: 'צירוף תמונה', onclick: () => chooser.click() }, icon('image')),
    text,
    el('button', { className: 'chat-send', type: 'submit', ariaLabel: 'שליחה', title: coarse ? 'שליחה' : ui.prefs().enterMessages ? 'שליחה (Enter). שורה חדשה: Shift+Enter' : 'שליחה (Ctrl+Enter)' }, icon('send')),
    error);
    const closed = el('p', { className: 'chat-closed' });
    const foot = el('div', { className: 'chat-foot' });
    function showForm() {
      closed.replaceChildren(...(list.length
        ? [`${other.name} כבר לא ברשימת החברים שלך, אז אי אפשר לכתוב כאן. מה שנכתב נשאר.`]
        : ['הודעות כותבים רק לחברים. ', link(`/u/${other.id}`, `לדף של ${other.name}`)]));
      foot.replaceChildren(canSend ? form : closed);
    }
    showForm();

    const node = el('section', { className: 'card chat' },
      el('header', { className: 'chat-head' },
        el('a', {
          className: 'chat-back',
          href: '/messages',
          ariaLabel: 'חזרה לרשימת השיחות',
          title: 'חזרה לרשימת השיחות',
          onclick: (e) => {
            e.preventDefault();
            back('/messages');
          },
        }, icon('right')),
        link(`/u/${other.id}`, avatar(other, { small: true }), el('span', { className: 'chat-name', dir: 'auto' }, other.name))),
      log, foot);
    render();

    // On a phone the conversation fills the screen, and has to shrink when the keyboard opens.
    // The keyboard covers the page without resizing it, so the visible part is measured here.
    const view = window.visualViewport;
    const fit = () => {
      node.style.setProperty('--view-h', `${view.height}px`);
      node.style.setProperty('--view-top', `${view.offsetTop}px`);
      if (pinned) toBottom();
    };
    view?.addEventListener('resize', fit);
    view?.addEventListener('scroll', fit);

    const onVisible = () => {
      if (document.hidden) {
        clearTimeout(timer);
        return;
      }
      lastActivity = Date.now();
      seen();
      schedule(0);
    };
    document.addEventListener('visibilitychange', onVisible);

    return {
      node,
      shown() {
        if (view) fit();
        grow();
        if (firstUnread && unreadMark.isConnected) log.scrollTop = Math.max(0, unreadMark.offsetTop - 12);
        else toBottom();
        pinned = log.scrollHeight - log.scrollTop - log.clientHeight < BOTTOM_SLACK;
        if (canSend && !coarse) text.focus();
        seen();
        schedule();
      },
      leave() {
        stopped = true;
        clearTimeout(timer);
        document.removeEventListener('visibilitychange', onVisible);
        view?.removeEventListener('resize', fit);
        view?.removeEventListener('scroll', fit);
      },
    };
  }

  return {
    node,
    shown: () => chat?.shown(),
    leave() {
      chat?.leave();
      document.removeEventListener('unread', reload);
    },
  };
}
