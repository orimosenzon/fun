// Feelings about a post or a comment. Five of them are drawn here: a heart, and four oranges
// from the orchard on the logo, each with a face. Any other emoji can be chosen as well.
// The drawings are plain SVG lit from the top left, and they move only while someone is
// choosing, or for a moment after a choice. Nothing here changes the order of the feed.

const SVG = 'http://www.w3.org/2000/svg';

// In the order they are offered, from the warmest to the angriest.
export const FEELINGS = [
  { kind: 'love', label: 'אהבתי' },
  { kind: 'funny', label: 'מצחיק' },
  { kind: 'surprise', label: 'הפתעה' },
  { kind: 'bored', label: 'משעמם' },
  { kind: 'anger', label: 'כעס' },
];
const LABELS = Object.fromEntries(FEELINGS.map((feeling) => [feeling.kind, feeling.label]));

export const feelingLabel = (kind) => LABELS[kind] ?? kind;

function s(tag, attributes = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  node.append(...children.filter(Boolean));
  return node;
}

const stops = (...list) => list.map(([offset, color, opacity]) => s('stop', {
  offset,
  'stop-color': color,
  ...(opacity === undefined ? {} : { 'stop-opacity': opacity }),
}));

// The paints are defined once for the whole page, and every drawing points at them.
function paints() {
  if (document.getElementById('rx-paints')) return;
  const radial = (id, box, ...list) => s('radialGradient', { id, ...box }, ...stops(...list));
  const lit = { cx: '0.36', cy: '0.28', r: '0.8' };
  document.body.append(s('svg', { id: 'rx-paints', class: 'rx-paints', 'aria-hidden': 'true', focusable: 'false' }, s('defs', {},
    radial('rx-peel', lit, ['0', '#ffe9a3'], ['0.3', '#ffc04a'], ['0.72', '#f4901f'], ['1', '#cf6410']),
    radial('rx-peel-anger', lit, ['0', '#ffcf8f'], ['0.3', '#fb8f3c'], ['0.72', '#e8581f'], ['1', '#b0340f']),
    radial('rx-peel-bored', lit, ['0', '#f8e2a4'], ['0.3', '#f1c26c'], ['0.72', '#e09c44'], ['1', '#b07226']),
    radial('rx-heart', { cx: '0.33', cy: '0.26', r: '0.85' }, ['0', '#ffab9c'], ['0.28', '#f6605a'], ['0.7', '#d92c49'], ['1', '#9b1238']),
    // The side away from the light, and the light that the table throws back on the bottom edge.
    radial('rx-shade', { cx: '0.4', cy: '0.3', r: '0.78' }, ['0.62', '#5a1e00', 0], ['1', '#5a1e00', 0.42]),
    radial('rx-gloss', { cx: '0.5', cy: '0.5', r: '0.5' }, ['0', '#fff', 0.95], ['0.6', '#fff', 0.35], ['1', '#fff', 0]),
    radial('rx-ground', { cx: '0.5', cy: '0.5', r: '0.5' }, ['0', '#4a2a08', 0.38], ['1', '#4a2a08', 0]),
    radial('rx-blush', { cx: '0.5', cy: '0.5', r: '0.5' }, ['0', '#ff5a4d', 0.5], ['1', '#ff5a4d', 0]),
    radial('rx-heat', { cx: '0.5', cy: '0.1', r: '0.78' }, ['0', '#c4160f', 0.8], ['0.55', '#c4160f', 0.35], ['1', '#c4160f', 0]),
    radial('rx-puff', { cx: '0.4', cy: '0.35', r: '0.7' }, ['0', '#ffffff', 1], ['0.55', '#e9f0f3', 1], ['1', '#b4c5ce', 1]),
    s('linearGradient', { id: 'rx-leaf', x1: '0', y1: '0', x2: '1', y2: '1' }, ...stops(['0', '#b5dc7a'], ['0.55', '#6f9b4e'], ['1', '#3f6a30'])),
    s('linearGradient', { id: 'rx-tear', x1: '0', y1: '0', x2: '0', y2: '1' }, ...stops(['0', '#d8f1ff'], ['1', '#4fa8e4'])),
    s('linearGradient', { id: 'rx-mouth', x1: '0', y1: '0', x2: '0', y2: '1' }, ...stops(['0', '#4a120c'], ['1', '#7a2415'])))));
}

const INK = '#3d220c';
const line = (d, width = 3.6, more = {}) => s('path', { d, fill: 'none', stroke: INK, 'stroke-width': width, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', ...more });
const part = (name, ...children) => s('g', { class: `rx-${name}` }, ...children);

function eye(x, y, r = 3.9) {
  return part('eye', s('ellipse', { cx: x, cy: y, rx: r * 0.9, ry: r, fill: INK }),
    s('circle', { cx: x - r * 0.3, cy: y - r * 0.38, r: r * 0.32, fill: '#fff', opacity: '0.9' }));
}

const cheeks = (y = 62) => part('cheeks',
  s('ellipse', { cx: 25, cy: y, rx: 8, ry: 5.5, fill: 'url(#rx-blush)' }),
  s('ellipse', { cx: 75, cy: y, rx: 8, ry: 5.5, fill: 'url(#rx-blush)' }));

// An orange: the fruit, its shaded side, the pores of the peel, a shine, and a leaf on the stalk.
function orange(peel, leafTurn, ...face) {
  const pores = [[66, 30], [73, 38], [60, 24], [78, 50], [70, 46], [24, 70], [31, 78], [76, 64]]
    .map(([cx, cy]) => s('circle', { cx, cy, r: 0.9, fill: '#7a3a00', opacity: '0.16' }));
  return [
    s('ellipse', { class: 'rx-ground', cx: 50, cy: 94.5, rx: 27, ry: 4.5, fill: 'url(#rx-ground)' }),
    part('body',
      s('circle', { cx: 50, cy: 55, r: 37, fill: `url(#${peel})` }),
      s('circle', { cx: 50, cy: 55, r: 37, fill: 'url(#rx-shade)' }),
      ...pores,
      s('ellipse', { class: 'rx-gloss', cx: 33, cy: 35, rx: 13, ry: 7.5, fill: 'url(#rx-gloss)', transform: 'rotate(-34 33 35)' }),
      s('ellipse', { cx: 50, cy: 19.2, rx: 5, ry: 1.7, fill: '#8a4a12', opacity: '0.45' }),
      part('leaf',
        s('path', { d: 'M50 19.5 q-0.6 -3.5 0.4 -6.2', fill: 'none', stroke: '#5a3910', 'stroke-width': 2.4, 'stroke-linecap': 'round' }),
        s('g', { transform: `rotate(${leafTurn} 50.4 13.6)` },
          s('path', { d: 'M50.4 13.6 C55 4.5 66 2.5 74.5 6.5 C70.5 15.5 59 18.5 50.4 13.6 Z', fill: 'url(#rx-leaf)' }),
          s('path', { d: 'M50.4 13.6 C58 10 66 8 74.5 6.5', fill: 'none', stroke: '#35602a', 'stroke-width': 0.9, 'stroke-linecap': 'round', opacity: '0.75' }),
          s('path', { d: 'M55 9.4 C59 6.4 64 5.2 69 5.6', fill: 'none', stroke: '#fff', 'stroke-width': 1.1, 'stroke-linecap': 'round', opacity: '0.4' }))),
      ...face),
  ];
}

const DRAW = {
  love: () => [
    s('ellipse', { class: 'rx-ground', cx: 50, cy: 94.5, rx: 22, ry: 4, fill: 'url(#rx-ground)' }),
    part('body',
      s('path', { class: 'rx-heart', d: 'M50 88 C22 68 9 50 12.5 34 C16 18.5 36 13.5 50 29.5 C64 13.5 84 18.5 87.5 34 C91 50 78 68 50 88 Z', fill: 'url(#rx-heart)' }),
      s('path', { d: 'M50 88 C22 68 9 50 12.5 34 C16 18.5 36 13.5 50 29.5 C64 13.5 84 18.5 87.5 34 C91 50 78 68 50 88 Z', fill: 'url(#rx-shade)', opacity: '0.7' }),
      s('ellipse', { class: 'rx-gloss', cx: 30.5, cy: 33.5, rx: 11.5, ry: 6.8, fill: 'url(#rx-gloss)', transform: 'rotate(-38 30.5 33.5)' }),
      s('ellipse', { cx: 67, cy: 29, rx: 5, ry: 2.8, fill: 'url(#rx-gloss)', opacity: '0.55', transform: 'rotate(28 67 29)' })),
  ],
  funny: () => orange('rx-peel', -6,
    cheeks(63),
    part('eyes', line('M28 48 q7 -9.5 14 0', 3.8), line('M58 48 q7 -9.5 14 0', 3.8)),
    part('mouth',
      s('path', { d: 'M30.5 58.5 Q50 63.5 69.5 58.5 Q69 81 50 81 Q31 81 30.5 58.5 Z', fill: 'url(#rx-mouth)' }),
      s('path', { d: 'M33.2 59.8 Q50 64.3 66.8 59.8 L66 65.2 Q50 68.6 34 65.2 Z', fill: '#fffaf0' }),
      s('ellipse', { cx: 50, cy: 76, rx: 9.5, ry: 4.6, fill: '#ff8a76' })),
    part('tear rx-tear-a', s('path', { d: 'M20 49 q-6.5 6 -3.8 10.4 q3.6 3.6 6.8 -0.6 q1.6 -4 -3 -9.8 Z', fill: 'url(#rx-tear)' })),
    part('tear rx-tear-b', s('path', { d: 'M80 49 q6.5 6 3.8 10.4 q-3.6 3.6 -6.8 -0.6 q-1.6 -4 3 -9.8 Z', fill: 'url(#rx-tear)' }))),
  surprise: () => orange('rx-peel', -34,
    cheeks(64),
    part('brows', line('M27.5 35.5 q7 -6 13.5 -1.5', 3.2), line('M72.5 35.5 q-7 -6 -13.5 -1.5', 3.2)),
    part('eyes',
      s('circle', { cx: 36.5, cy: 50, r: 8, fill: '#fffdf6' }), s('circle', { cx: 63.5, cy: 50, r: 8, fill: '#fffdf6' }),
      eye(36.5, 50.5, 4.2), eye(63.5, 50.5, 4.2)),
    part('mouth',
      s('ellipse', { cx: 50, cy: 72, rx: 6.6, ry: 8.4, fill: 'url(#rx-mouth)' }),
      s('ellipse', { cx: 50, cy: 76.6, rx: 4, ry: 2.8, fill: '#ff8a76', opacity: '0.85' })),
    part('sparks',
      line('M16 23 l-5 -4.5', 2.6, { stroke: '#f4901f' }), line('M10.5 33.5 l-6.5 -1.5', 2.6, { stroke: '#f4901f' }),
      line('M84 23 l5 -4.5', 2.6, { stroke: '#f4901f' }), line('M89.5 33.5 l6.5 -1.5', 2.6, { stroke: '#f4901f' }))),
  bored: () => orange('rx-peel-bored', 40,
    part('eyes',
      s('circle', { cx: 36.5, cy: 51, r: 7.4, fill: '#fffdf6' }), s('circle', { cx: 63.5, cy: 51, r: 7.4, fill: '#fffdf6' }),
      part('pupils', eye(38.5, 53, 3.5), eye(65.5, 53, 3.5)),
      // Heavy lids, in the colour of the peel, cover the top half of each eye.
      part('lids',
        s('path', { d: 'M28.4 51.6 A8.2 8.2 0 0 1 44.6 51.6 Z', fill: '#e8b55e' }), s('path', { d: 'M55.4 51.6 A8.2 8.2 0 0 1 71.6 51.6 Z', fill: '#e8b55e' }),
        line('M28.2 51.6 h16.6', 2.8), line('M55.2 51.6 h16.6', 2.8))),
    part('brows', line('M28 41 h15', 2.6, { opacity: '0.7' }), line('M57 41 h15', 2.6, { opacity: '0.7' })),
    part('mouth', line('M40.5 72.5 q9.5 -1.2 19 0.6', 3.4))),
  anger: () => [
    part('steam rx-steam-a', s('circle', { cx: 15, cy: 24, r: 5.2, fill: 'url(#rx-puff)' }), s('circle', { cx: 9.5, cy: 19.5, r: 3.8, fill: 'url(#rx-puff)' }), s('circle', { cx: 16.5, cy: 17, r: 3.2, fill: 'url(#rx-puff)' })),
    part('steam rx-steam-b', s('circle', { cx: 85, cy: 24, r: 5.2, fill: 'url(#rx-puff)' }), s('circle', { cx: 90.5, cy: 19.5, r: 3.8, fill: 'url(#rx-puff)' }), s('circle', { cx: 83.5, cy: 17, r: 3.2, fill: 'url(#rx-puff)' })),
    ...orange('rx-peel-anger', 14,
      s('circle', { class: 'rx-heat', cx: 50, cy: 55, r: 37, fill: 'url(#rx-heat)' }),
      part('eyes', eye(37, 53.5, 4), eye(63, 53.5, 4)),
      part('brows', line('M25.5 40.5 L44 48', 4.6), line('M74.5 40.5 L56 48', 4.6)),
      part('mouth', line('M38 75 Q50 64.5 62 75', 3.8))),
  ],
};

// One of the five drawings. It stands still unless it sits inside .rx-live or .rx-once.
export function drawing(kind) {
  paints();
  return s('svg', { class: `rx-art rx-${kind}`, viewBox: '0 0 100 100', 'aria-hidden': 'true', focusable: 'false' }, ...DRAW[kind]());
}

// What stands for a feeling anywhere on the page: a drawing, or the emoji itself.
export function face(kind) {
  if (DRAW[kind]) return drawing(kind);
  const node = document.createElement('span');
  node.className = 'rx-emoji';
  node.textContent = kind;
  return node;
}

// ---------- Choosing a feeling, and seeing what others felt ----------

const RECENT_KEY = 'rx-recent';
const MAX_RECENT = 16;
const SHOWN_PEOPLE = 40;
let emojiList = null;

const firstName = (name) => name.trim().split(/\s+/)[0];
// "and" is one letter stuck to the next word, which reads badly before a name in Latin letters.
const and = (name) => (/^[֐-׿]/.test(name) ? `ו${name}` : `ו-${name}`);

function whoText(groups) {
  const people = groups.flatMap((group) => group.people);
  const total = groups.reduce((sum, group) => sum + group.count, 0);
  if (!people.length) return '';
  if (total === 1) return people[0].name;
  if (total === 2 && people.length === 2) return `${firstName(people[0].name)} ${and(firstName(people[1].name))}`;
  return `${firstName(people[0].name)}, ${firstName(people[1]?.name ?? '')} ועוד ${total - 2}`;
}

function recentEmoji(add) {
  try {
    const list = JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]').filter((item) => typeof item === 'string');
    if (!add) return list;
    const next = [add, ...list.filter((item) => item !== add)].slice(0, MAX_RECENT);
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
    return next;
  } catch {
    return [];
  }
}

// The whole standard set, by group, with a search in Hebrew. Loaded the first time it is opened.
function emojiPanel({ ui, onPick }) {
  const { el, icon } = ui;
  const box = el('dialog', { className: 'rx-all', 'aria-label': 'כל האימוג׳י' });
  const search = el('input', { type: 'search', placeholder: 'חיפוש: לב, עוגה, כלב…', 'aria-label': 'חיפוש אימוג׳י', enterKeyHint: 'search', dir: 'auto' });
  const tabs = el('nav', { className: 'rx-tabs', 'aria-label': 'קבוצות' });
  const list = el('div', { className: 'rx-list' }, el('p', { className: 'muted rx-wait' }, 'רגע…'));
  const close = () => {
    box.close();
    box.remove();
  };
  box.append(el('div', { className: 'rx-all-body' },
    el('div', { className: 'rx-all-head' }, search,
      el('button', { className: 'icon-button', type: 'button', 'aria-label': 'סגירה', onclick: close }, icon('close'))),
    tabs, list));
  box.addEventListener('click', (e) => {
    if (e.target === box) return close();
    const chosen = e.target.closest('[data-emoji]')?.dataset.emoji;
    if (!chosen) return;
    recentEmoji(chosen);
    close();
    onPick(chosen);
  });
  box.addEventListener('cancel', (e) => {
    e.preventDefault();
    close();
  });

  const key = ([emoji, name]) => el('button', { className: 'rx-key', type: 'button', title: name, 'aria-label': name, 'data-emoji': emoji }, emoji);
  const section = (name, items) => el('section', {}, el('h3', {}, name), el('div', { className: 'rx-keys' }, ...items.map(key)));

  function show(groups) {
    const names = new Map(groups.flatMap((group) => group.emoji.map((item) => [item[0], item[1]])));
    const recent = recentEmoji().filter((emoji) => names.has(emoji)).map((emoji) => [emoji, names.get(emoji)]);
    const sections = groups.map((group) => section(group.name, group.emoji));
    const browse = [recent.length ? section('לאחרונה', recent) : null, ...sections].filter(Boolean);
    tabs.replaceChildren(...groups.map((group, at) => el('button', {
      className: 'rx-tab',
      type: 'button',
      title: group.name,
      'aria-label': group.name,
      onclick: () => {
        search.value = '';
        list.replaceChildren(...browse);
        sections[at].scrollIntoView({ block: 'start' });
      },
    }, group.icon)));
    list.replaceChildren(...browse);
    search.addEventListener('input', () => {
      const asked = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!asked.length) return list.replaceChildren(...browse);
      const found = groups.flatMap((group) => group.emoji).filter((item) => {
        const text = item.join(' ').toLowerCase();
        return asked.every((word) => text.includes(word));
      });
      list.replaceChildren(found.length ? section(`נמצאו ${found.length}`, found.slice(0, 240)) : el('p', { className: 'muted rx-wait' }, 'לא נמצא אימוג׳י כזה.'));
      list.scrollTop = 0;
    });
  }

  // A dialog puts the cursor in its first field as it opens. On a phone that raises the keyboard
  // over half the list, so there the list itself takes the focus.
  if (matchMedia('(hover: hover)').matches) search.setAttribute('autofocus', '');
  else {
    list.tabIndex = -1;
    list.setAttribute('autofocus', '');
  }
  document.body.append(box);
  box.showModal();
  (emojiList ??= fetch('/emoji.json').then((response) => {
    if (!response.ok) throw new Error('emoji list');
    return response.json();
  })).then(({ groups }) => show(groups)).catch(() => {
    emojiList = null;
    list.replaceChildren(el('p', { className: 'error rx-wait' }, 'הרשימה לא נטענה, נסו שוב.'));
  });
}

// The feelings of one post or comment. Returns the button that opens the choice (wrapped, so
// the choice can open beside it) and the line that shows what was felt and by whom.
export function feelings({ path, reactions, me, ui, small = false }) {
  const { el, icon, api, tell, go, link } = ui;
  let groups = reactions;
  let picker = null;
  const button = el('button', { className: 'plain rx-button', type: 'button', 'aria-haspopup': 'true', 'aria-expanded': 'false' });
  const holder = el('span', { className: `rx-holder${small ? ' small' : ''}` }, button);
  const summary = el('div', { className: `rx-summary${small ? ' small' : ''}` });
  const mine = () => groups.find((group) => group.mine)?.kind ?? null;

  function chip(group) {
    return el('button', {
      className: `rx-chip${group.mine ? ' mine' : ''}`,
      type: 'button',
      title: [feelingLabel(group.kind), ...group.people.slice(0, 12).map((person) => person.name)].join('\n'),
      'aria-label': `${feelingLabel(group.kind)}: ${group.count}${group.mine ? ', כולל אותך' : ''}`,
      'aria-pressed': String(group.mine),
      'data-kind': group.kind,
      onclick: () => (me ? choose(group.kind) : go('/login')),
    }, el('span', { className: 'rx-face' }, face(group.kind)), el('span', { className: 'rx-count' }, String(group.count)));
  }

  function render() {
    const kind = mine();
    button.replaceChildren(
      kind ? el('span', { className: 'rx-face' }, face(kind)) : icon('feel'),
      el('span', { className: 'label' }, kind ? LABELS[kind] ?? 'הרגש שלי' : 'רגש'));
    button.className = `plain rx-button${kind ? ` chosen rx-is-${LABELS[kind] ? kind : 'emoji'}` : ''}`;
    button.setAttribute('aria-label', kind ? `הרגש שלך: ${feelingLabel(kind)}. לחיצה משנה` : 'הבעת רגש');
    const text = whoText(groups);
    summary.replaceChildren(...groups.map(chip),
      ...(text ? [el('button', { className: 'plain rx-who', type: 'button', title: 'מי הרגיש מה', onclick: showPeople }, text)] : []));
    summary.hidden = !groups.length;
  }

  // What the page shows at once, before the server answers.
  function withMine(kind) {
    const self = { id: me.id, name: me.name };
    const next = groups
      .map((group) => (group.mine ? { ...group, mine: false, count: group.count - 1, people: group.people.filter((person) => person.id !== me.id) } : group))
      .filter((group) => group.count > 0);
    if (!kind) return next;
    const same = next.find((group) => group.kind === kind);
    if (!same) return [...next, { kind, count: 1, mine: true, people: [self] }];
    return next.map((group) => (group === same ? { ...group, mine: true, count: group.count + 1, people: [...group.people, self] } : group));
  }

  // The chosen feeling rises from the button, plays its scene once, and fades.
  function celebrate(kind) {
    const pop = el('span', { className: 'rx-pop rx-once', 'aria-hidden': 'true' }, face(kind));
    holder.append(pop);
    pop.addEventListener('animationend', (e) => e.target === pop && pop.remove());
    setTimeout(() => pop.remove(), 3000);
    summary.querySelector('.rx-chip.mine')?.classList.add('bump');
  }

  async function choose(kind) {
    close();
    const before = groups;
    const next = mine() === kind ? null : kind;
    groups = withMine(next);
    render();
    if (next) celebrate(next);
    try {
      const { reactions: saved } = await api(path, next ? { method: 'PUT', body: { kind: next } } : { method: 'DELETE' });
      // Redrawing would cut the little scene short, so it happens only when the server saw it differently.
      const same = JSON.stringify(saved.map((group) => [group.kind, group.count, group.mine])) === JSON.stringify(groups.map((group) => [group.kind, group.count, group.mine]));
      groups = saved;
      if (!same) render();
    } catch (err) {
      groups = before;
      render();
      tell(err.message);
    }
  }

  function showPeople() {
    const box = el('dialog', { className: 'dialog rx-people' });
    const shut = () => {
      box.close();
      box.remove();
    };
    box.append(el('div', { className: 'dialog-body' },
      el('h2', {}, 'מי הרגיש מה'),
      el('div', { className: 'rx-people-list' }, ...groups.map((group) => el('div', { className: 'rx-people-group' },
        el('span', { className: 'rx-face' }, face(group.kind)),
        el('ul', {},
          ...group.people.slice(0, SHOWN_PEOPLE).map((person) => el('li', { dir: 'auto' }, link(`/u/${person.id}`, person.name))),
          group.count > Math.min(group.people.length, SHOWN_PEOPLE) ? el('li', { className: 'muted' }, `ועוד ${group.count - Math.min(group.people.length, SHOWN_PEOPLE)}`) : null)))),
      el('div', { className: 'dialog-actions' }, el('button', { className: 'ghost', type: 'button', onclick: shut }, 'סגירה'))));
    box.addEventListener('click', (e) => (e.target === box || e.target.closest('a')) && shut());
    box.addEventListener('cancel', (e) => {
      e.preventDefault();
      shut();
    });
    document.body.append(box);
    box.showModal();
  }

  const outside = (e) => !holder.contains(e.target) && close();
  const onKey = (e) => {
    if (e.key !== 'Escape') return;
    e.stopPropagation();
    close();
    button.focus();
  };

  function close() {
    if (!picker) return;
    picker.remove();
    picker = null;
    button.setAttribute('aria-expanded', 'false');
    holder.closest('.card, .bubble')?.classList.remove('rx-open');
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', onKey, true);
  }

  function open(byKeyboard) {
    const current = mine();
    const picks = FEELINGS.map(({ kind, label }, at) => {
      const chosen = current === kind;
      const pick = el('button', {
        className: `rx-pick${chosen ? ' chosen' : ''}`,
        type: 'button',
        role: 'menuitem',
        'aria-label': chosen ? `${label}, הבחירה שלך. לחיצה מבטלת אותה` : label,
        onclick: () => choose(kind),
      }, el('span', { className: 'rx-pick-art' }, drawing(kind)), el('span', { className: 'rx-pick-label' }, label));
      pick.style.setProperty('--rx-wait', `${at * 55}ms`);
      return pick;
    });
    const more = el('button', {
      className: 'rx-pick rx-more',
      type: 'button',
      role: 'menuitem',
      'aria-label': 'אימוג׳י אחר',
      onclick: () => {
        close();
        emojiPanel({ ui, onPick: choose });
      },
    }, el('span', { className: 'rx-pick-art' }, icon('plus')), el('span', { className: 'rx-pick-label' }, 'אחר'));
    more.style.setProperty('--rx-wait', `${FEELINGS.length * 55}ms`);
    // It opens above the button, moved sideways when it would leave the screen, and below when
    // the top of the page is too close. It is measured out of the way first: on a phone, a box
    // that sticks out of the page for a moment makes the whole page jump.
    picker = el('div', { className: 'rx-picker rx-once measuring', role: 'menu', 'aria-label': 'בחירת רגש' }, ...picks, more);
    holder.append(picker);
    const edge = 8;
    const gap = 10;
    const { width, height } = picker.getBoundingClientRect();
    const at = holder.getBoundingClientRect();
    const left = getComputedStyle(holder).direction === 'rtl' ? at.right + gap - width : at.left - gap;
    const shift = Math.max(edge, Math.min(left, document.documentElement.clientWidth - edge - width)) - left;
    picker.style.setProperty('--rx-shift', `${shift}px`);
    const bar = document.querySelector('.topbar');
    const roof = bar && getComputedStyle(bar).position !== 'static' ? bar.getBoundingClientRect().bottom : 0;
    if (at.top - edge - height < roof + edge) picker.classList.add('below');
    picker.classList.remove('measuring');
    holder.closest('.card, .bubble')?.classList.add('rx-open');

    button.setAttribute('aria-expanded', 'true');
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', onKey, true);
    if (byKeyboard) (picker.querySelector('.chosen') ?? picks[0]).focus();
  }

  button.addEventListener('click', (e) => {
    if (!me) return go('/login');
    if (picker) return close();
    // A click from the keyboard has no pointer behind it, and then the choice takes the focus.
    open(e.detail === 0);
  });
  render();
  return { button: holder, summary };
}
