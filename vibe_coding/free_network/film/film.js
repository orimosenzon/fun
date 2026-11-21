// The explanation film, drawn in the page. Every picture is a pure function of the time, so
// film/render.py can photograph it frame by frame, and a browser can simply play it.
// The words come from script.json, their timing from audio/timing.json (film/narrate.py),
// and the screens of the network from shots/ (film/shoot.py).
const LEAD = 0.5;
const TAIL = 0.7;
const FADE = 0.45;
const PORT_W = 1380;
const PORT_H = 873;
const PORT_X = 270;
const PORT_Y = 74;

const json = (path) => fetch(path).then((response) => response.json());
const [script, timing, stills] = await Promise.all([json('script.json'), json('audio/timing.json'), json('shots/marks.json')]);
const { marks, sizes } = stills;
const stage = document.getElementById('stage');

const ICONS = {
  users: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 3a4 4 0 1 0 0 8a4 4 0 1 0 0-8M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75',
  home: 'M3 11l9-8 9 8M5 10v10h14V10',
  comment: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
  eyeOff: 'M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M1 1l22 22',
  check: 'M20 6L9 17l-5-5',
  leaf: 'M11 20A7 7 0 0 1 4 13C4 7 10 4 20 4c0 10-3 16-9 16zM4 21c2-6 6-9 11-11',
  shield: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
  code: 'M16 18l6-6-6-6M8 6l-6 6 6 6',
  megaphoneOff: 'M3 11v2a1 1 0 0 0 1 1h3l5 4V6L7 10H4a1 1 0 0 0-1 1zM16 9l5 6M21 9l-5 6',
};

function h(tag, className = '', ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function svg(viewBox, className, inner) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  node.setAttribute('viewBox', viewBox);
  if (className) node.setAttribute('class', className);
  node.innerHTML = inner;
  return node;
}

const icon = (name) => svg('0 0 24 24', 'line', `<path d="${ICONS[name]}"/>`);
const crown = (className) => svg('0 0 120 84', className,
  '<path d="M8 74 L4 22 L36 46 L60 8 L84 46 L116 22 L112 74 Z" fill="#f2c14e" stroke="#8a5a12" stroke-width="5" stroke-linejoin="round"/><circle cx="60" cy="8" r="7" fill="#e2674f"/><circle cx="4" cy="22" r="6" fill="#e2674f"/><circle cx="116" cy="22" r="6" fill="#e2674f"/>');

function image(src, className = '') {
  const node = h('img', className);
  node.src = src;
  return node;
}

// The picture the network draws from a first name, as in public/app.js.
function face(name, color, size = 112) {
  const letters = [...name];
  const node = h('span', 'face-art', name);
  node.style.background = `var(--a${color})`;
  node.style.width = `${size}px`;
  node.style.height = `${size}px`;
  node.style.fontSize = `${Math.min(0.5, 0.78 / (0.62 * letters.length)) * size}px`;
  return node;
}

const clamp = (x) => Math.max(0, Math.min(1, x));
const smooth = (x) => clamp(x) ** 2 * (3 - 2 * clamp(x));
const settle = (x) => 1 - (1 - clamp(x)) ** 3;
const spring = (x) => 1 + 2.6 * (clamp(x) - 1) ** 3 + 1.6 * (clamp(x) - 1) ** 2;
const ramp = (t, at, length = 0.5) => clamp((t - at) / length);
// Visible from `at` until `until`, fading at both ends.
const during = (t, at, until = Infinity, length = 0.3) => ramp(t, at, length) * (1 - ramp(t, until, length));

function pop(node, t, at, { until = Infinity, from = 0.5, length = 0.5 } = {}) {
  node.style.opacity = during(t, at, until);
  node.style.transform = `scale(${from + (1 - from) * spring(ramp(t, at, length))})`;
}

function rise(node, t, at, { until = Infinity, by = 40, length = 0.6 } = {}) {
  node.style.opacity = during(t, at, until, 0.35);
  node.style.transform = `translateY(${by * (1 - settle(ramp(t, at, length)))}px)`;
}

function corners(scene) {
  for (const [name, style] of [['top-right', 'right:0;top:0;height:420px'], ['bottom-left', 'left:0;bottom:0;height:400px']]) {
    const art = image(`../public/art/${name}.svg`, 'corner');
    art.style.cssText = style;
    scene.append(art);
  }
}

// A browser window showing one or more stills of the same size. A view is the part of the
// still that fills the window: its corner and its width, in still pixels.
function browser(scene, names) {
  const [width, height] = sizes[names[0]];
  const sheet = h('div', 'sheet');
  sheet.style.width = `${width}px`;
  sheet.style.height = `${height}px`;
  const pictures = {};
  for (const name of names) {
    pictures[name] = image(`shots/${name}.png`);
    sheet.append(pictures[name]);
  }
  const node = h('div', 'browser', h('div', 'bar', h('i'), h('i'), h('i'), h('span', '', 'הרשת החופשית')), h('div', 'port', sheet));
  scene.append(node);
  let current = { x: 0, y: 0, w: width };
  return {
    node,
    view(next) {
      current = next;
      const scale = PORT_W / next.w;
      sheet.style.transform = `translate(${-next.x * scale}px, ${-next.y * scale}px) scale(${scale})`;
    },
    show(name, opacity = 1) {
      pictures[name].style.opacity = opacity;
    },
    ring([x, y, w, hgt], pad = 16, className = 'ring') {
      const ring = h('div', className);
      ring.style.cssText = `left:${x - pad}px;top:${y - pad}px;width:${w + 2 * pad}px;height:${hgt + 2 * pad}px`;
      sheet.append(ring);
      return ring;
    },
    put(node_, x, y) {
      node_.style.left = `${x}px`;
      node_.style.top = `${y}px`;
      sheet.append(node_);
      return node_;
    },
    // Where a point of the still is on the stage, in the current view.
    onStage(x, y) {
      const scale = PORT_W / current.w;
      return [PORT_X + (x - current.x) * scale, PORT_Y + (y - current.y) * scale];
    },
  };
}

// Travels between views: [[time, view], ...]. Between two times the view glides.
function travel(t, stops) {
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
  const [t0, a] = stops[i];
  const [t1, b] = stops[i + 1] ?? stops[i];
  const p = smooth((t - t0) / Math.max(0.001, t1 - t0));
  return { x: a.x + (b.x - a.x) * p, y: a.y + (b.y - a.y) * p, w: a.w + (b.w - a.w) * p };
}

function cursor(scene) {
  const node = svg('0 0 24 24', 'cursor', '<path d="M5 3l14 8-6 2-3 6z" fill="#0c2330" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/>');
  node.style.position = 'absolute';
  scene.append(node);
  return node;
}

function moveCursor(node, t, stops) {
  let i = 0;
  while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
  const [t0, x0, y0] = stops[i];
  const [t1, x1, y1] = stops[i + 1] ?? stops[i];
  const p = smooth((t - t0) / Math.max(0.001, t1 - t0));
  node.style.left = `${x0 + (x1 - x0) * p}px`;
  node.style.top = `${y0 + (y1 - y0) * p}px`;
}

const SCENES = {
  // "Today you can choose, democratically, your tyrant."
  motto(scene, { line, word }) {
    scene.classList.add('night');
    const words = ['כיום', 'אתה', 'יכול', 'לבחור,', null, 'באופן', 'דמוקרטי,', null, 'את', 'העריץ', 'שלך.'];
    const top = crown('crown');
    const spans = [];
    const motto = h('div', 'motto center', ...words.map((text) => {
      if (!text) return h('br');
      const span = h('span', text.startsWith('העריץ') ? 'w tyrant' : 'w', text);
      if (text.startsWith('העריץ')) span.append(top);
      spans.push([span, text.replace(/[.,]/g, '')]);
      return span;
    }));
    const shapes = [
      ['#3d6fa8', '<circle cx="48" cy="48" r="30" fill="none" stroke="#fff" stroke-width="12"/>'],
      ['#b5476b', '<path d="M48 14 84 78H12z" fill="#fff"/>'],
      ['#20262b', '<path d="M16 22h64M16 48h64M16 74h40" stroke="#fff" stroke-width="12" stroke-linecap="round"/>'],
      ['#6f58a8', '<rect x="18" y="18" width="60" height="60" rx="16" fill="#fff"/>'],
      ['#3f7d6a', '<path d="M12 60c12-36 24-36 36 0s24 36 36 0" fill="none" stroke="#fff" stroke-width="12" stroke-linecap="round"/>'],
    ];
    const tiles = shapes.map(([color, glyph]) => {
      const tile = h('div', 'tile', svg('0 0 96 96', 'glyph', glyph), crown('mini-crown'), h('span', 'ltd', 'חברה מסחרית'));
      tile.style.background = color;
      return tile;
    });
    const picker = h('div', 'picker', h('span', 'tick', svg('0 0 24 24', '', `<path d="${ICONS.check}"/>`)));
    const sell = h('span', 'sell', 'למכור');
    const strike = h('i');
    const good = h('span', 'good', 'טובת המשתמשים', strike);
    const what = h('span', '', 'האינטרס:');
    scene.append(motto, h('div', 'tiles center', ...tiles), picker, h('div', 'interest center', what, sell, good));
    const tileX = (i) => 960 - (5 * 190 + 4 * 56) / 2 + i * (190 + 56) - 18;

    return (t) => {
      for (const [span, text] of spans) rise(span, t, word(0, text) - 0.12, { by: 26, length: 0.4 });
      pop(top, t, line(0).end - 0.5, { from: 0.2, length: 0.6 });
      const up = smooth(ramp(t, line(1).start - 0.3, 0.8));
      motto.style.transform = `translateY(${-190 * up}px) scale(${1 - 0.52 * up})`;
      tiles.forEach((tile, i) => pop(tile, t, line(1).start + 0.25 + i * 0.16));
      // The choice hops between the networks and lands on one: "everyone chooses for themselves".
      const hops = [1, 3, 0, 2];
      const from = word(1, 'וכל');
      const step = Math.min(hops.length - 1, Math.max(0, Math.floor((t - from) / 0.42)));
      picker.style.left = `${tileX(hops[step])}px`;
      picker.style.opacity = during(t, from - 0.1, line(3).start - 0.2);
      pop(picker.firstChild, t, from + 0.42 * hops.length);
      tiles.forEach((tile, i) => {
        rise(tile.querySelector('.ltd'), t, line(2).start + 0.5 + i * 0.22, { by: -14 });
        const drop = settle(ramp(t, line(2).start + 0.9 + i * 0.22, 0.5));
        const mini = tile.querySelector('.mini-crown');
        mini.style.opacity = drop;
        mini.style.transform = `translateY(${-70 * (1 - drop)}px) rotate(${-10 + 10 * drop}deg)`;
      });
      rise(what, t, line(3).start + 0.1);
      pop(sell, t, word(3, 'למכור') - 0.1);
      rise(good, t, word(3, 'טובת') - 0.15);
      strike.style.transform = `scaleX(${settle(ramp(t, word(3, 'המשתמשים') + 0.35, 0.5))})`;
    };
  },

  // Whoever needs to profit from the place decides what happens in it.
  problem(scene, { line, word }) {
    corners(scene);
    const uses = [['comment', 'מדברים עם חברים', 'מדברים'], ['home', 'שומעים מה קורה בשכונה', 'שומעים'], ['users', 'מתארגנים יחד', 'ומתארגנים']]
      .map(([name, text, cue]) => [h('div', 'use', h('span', 'dot', icon(name)), text), cue]);
    const owner = h('div', 'owner', h('span', '', h('b', '', '₪'), 'מנוהל בידי מי שצריך להרוויח'));
    const colors = [1, 2, 3, 5];
    const posts = colors.map((color, i) => {
      const dot = h('span', 'face');
      dot.style.background = `var(--a${color})`;
      const bars = h('span', 'bars', ...[1, 0.8, 0.55].map((width) => {
        const bar = h('i');
        bar.style.width = `${width * 100}%`;
        return bar;
      }));
      const post = h('div', 'mpost', dot, bars, h('span', 'tag', 'ממומן'), h('span', 'hidden-eye', icon('eyeOff')));
      post.dataset.slot = i;
      return post;
    });
    const mock = h('div', 'mock', owner, ...posts);
    const decides = [['מי יראה', 'את מה שכתבת', 'מי'], ['באיזה סדר', 'יופיעו התגובות', 'באיזה'], ['מה יתפוס', 'את תשומת הלב שלך', 'ומה']]
      .map(([title, rest, cue]) => [h('div', 'decide', title, h('small', '', rest)), cue]);
    scene.append(h('div', 'uses', ...uses.map(([node]) => node)), mock, h('div', 'decides', ...decides.map(([node]) => node)));
    const slotY = (slot) => 34 + slot * 176;

    return (t) => {
      mock.style.opacity = during(t, 0.15);
      mock.style.transform = `translateY(${50 * (1 - settle(ramp(t, 0.15, 0.8)))}px)`;
      for (const [node, cue] of uses) {
        rise(node, t, word(0, cue) - 0.1);
        node.style.opacity *= 1 - 0.6 * ramp(t, line(1).start - 0.2, 0.5);
      }
      pop(owner, t, word(0, 'מנוהל') - 0.1, { from: 0.7 });
      const times = decides.map(([node, cue]) => word(1, cue) - 0.1);
      decides.forEach(([node], i) => {
        rise(node, t, times[i], { by: 24 });
        const lit = during(t, times[i], times[i + 1] ?? Infinity, 0.25);
        node.style.borderInlineStartColor = `rgba(194, 100, 26, ${lit})`;
        node.style.background = `rgba(252, 246, 234, ${0.6 + 0.4 * lit})`;
      });
      // Who will see: one post fades away. In what order: two swap. What grabs you: one is pushed.
      const hide = smooth(ramp(t, times[0] + 0.5, 0.7));
      const swap = smooth(ramp(t, times[1] + 0.4, 0.9));
      const push = smooth(ramp(t, times[2] + 0.4, 0.7));
      const order = [0, 1 + 2 * swap, 2, 3 - 2 * swap];
      posts.forEach((post, i) => {
        const lift = i === 3 || i === 1 ? Math.sin(Math.PI * swap) * (i === 1 ? 30 : -30) : 0;
        post.style.top = `${slotY(order[i])}px`;
        post.style.transform = `translateX(${lift}px) scale(${i === 2 ? 1 + 0.07 * push : 1})`;
        post.style.zIndex = i === 2 ? 2 : 1;
      });
      posts[0].style.opacity = 1 - 0.78 * hide;
      posts[0].querySelector('.hidden-eye').style.opacity = hide;
      posts[2].querySelector('.tag').style.opacity = push;
      posts[2].style.borderColor = push > 0.5 ? '#f2c14e' : '#ecdcc0';
      posts[2].style.boxShadow = `0 0 0 ${10 * push}px rgba(242, 193, 78, 0.35)`;
    };
  },

  // Too important for that: a network run like Wikipedia.
  idea(scene, { line, word }) {
    corners(scene);
    const big = h('div', 'big center', 'רשת חברתית היא דבר חשוב מדי', h('small', '', 'בשביל זה.'));
    const logo = image('../public/favicon.svg', 'logo');
    const name = h('div', 'name center', 'הרשת החופשית');
    const like = h('div', 'like center', 'הרעיון דומה לוויקיפדיה: ', h('b', '', 'פרויקט ללא מטרות רווח'), h('br'), 'שפועל לטובת מי שמשתמש בו.');
    const badges = [['megaphoneOff', 'בלי פרסומות', 'בלי'], ['shield', 'בלי מעקב', 'מעקב'], ['code', 'קוד פתוח', 'קוד']]
      .map(([glyph, text, cue]) => [h('div', 'badge', h('span', 'dot', icon(glyph)), text), cue]);
    scene.append(big, logo, name, like, h('div', 'badges center', ...badges.map(([node]) => node)));
    return (t) => {
      rise(big, t, 0.2, { until: line(1).start + 0.15 });
      rise(big.querySelector('small'), t, word(0, 'בשביל') - 0.1, { until: line(1).start + 0.15, by: 20 });
      pop(logo, t, line(1).start + 0.55, { from: 0.3, length: 0.7 });
      rise(name, t, word(1, 'הרשת') - 0.15);
      rise(like, t, line(2).start);
      for (const [node, cue] of badges) pop(node, t, word(3, cue) - 0.15, { from: 0.7 });
    };
  },

  // You join by invitation: an email, a link, a name and a picture.
  join(scene, { line, word }) {
    const invited = h('div', 'invited center', 'מצטרפים בהזמנה');
    const sender = face('אורי', 0);
    const linkLine = h('span', 'linkline', 'הקישור האישי שלך לכניסה');
    const mail = h('div', 'mail',
      h('div', 'from', sender, h('div', '', h('strong', '', 'אורי מוסנזון'), h('small', '', 'אל: דנה'))),
      h('h3', '', 'הזמנה אישית לרשת החופשית'),
      h('p', '', 'היי דנה,'),
      h('p', '', 'אני בונה רשת חברתית חדשה. אין בה פרסומות, ואין בה אלגוריתם שמחליט בשבילנו מה נראה. אשמח לראות אותך בין הראשונים.'),
      linkLine);
    scene.append(invited, mail);
    const window_ = browser(scene, ['welcome']);
    window_.show('welcome');
    const m = marks.welcome;
    const nameRing = window_.ring(m.name, 14);
    const faceRing = window_.ring(m.face, 22);
    faceRing.style.borderRadius = '50%';
    const friends = [['אורי', 0], ['גיל', 0], ['מאיה', 5], ['יעל', 1], ['תמר', 3], ['רון', 4]].map(([text, color]) => face(text, color, 124));
    const drawn = h('div', 'panel', h('h4', '', 'אין תמונה? הרשת מציירת אחת מהשם הפרטי'), h('div', 'faces', ...friends));
    drawn.style.top = '610px';
    scene.append(drawn);
    const pointer = cursor(scene);
    const click = word(1, 'לוחצים') + 0.55;
    const wide = { x: 500, y: 40, w: 1720 };
    const close = { x: 900, y: 300, w: 1100 };

    return (t) => {
      rise(invited, t, 0.15, { until: line(1).start - 0.3 });
      mail.style.opacity = during(t, line(1).start - 0.1, click + 0.25);
      mail.style.transform = `translateY(${40 * (1 - settle(ramp(t, line(1).start - 0.1, 0.6)))}px) scale(${1 + 0.05 * smooth(ramp(t, click, 0.4))})`;
      linkLine.style.background = t > click - 0.05 && t < click + 0.3 ? '#cfe1ee' : '#eaf1f5';
      pointer.style.opacity = during(t, line(1).start + 0.8, click + 0.3, 0.2);
      moveCursor(pointer, t, [[line(1).start + 0.8, 1250, 900], [click - 0.25, 1230, 648]]);
      pointer.style.transform = `scale(${1 - 0.18 * during(t, click - 0.08, click + 0.1, 0.08)})`;
      window_.node.style.opacity = ramp(t, click + 0.15, 0.4);
      window_.node.style.transform = `scale(${0.94 + 0.06 * settle(ramp(t, click + 0.15, 0.6))})`;
      const zoom = word(2, 'ואם') - 0.3;
      window_.view(travel(t, [[click, wide], [zoom, wide], [zoom + 1, close]]));
      nameRing.style.opacity = during(t, word(2, 'שם') - 0.15, word(2, 'ותמונה') - 0.05, 0.2);
      faceRing.style.opacity = during(t, word(2, 'ותמונה') - 0.05, zoom + 0.2, 0.2);
      rise(drawn, t, word(2, 'הרשת') - 0.2, { by: 20 });
      friends.forEach((node, i) => pop(node, t, word(2, 'מציירת') + i * 0.14));
    };
  },

  // The feed, and writing a post: who will see it, and how many it reaches.
  feed(scene, { line, word }) {
    const window_ = browser(scene, ['feed', 'compose', 'compose_public']);
    const { feed, compose } = marks;
    const rule = window_.ring(feed.rule, 16);
    const audience = window_.ring(compose.audience, 14);
    const toFriends = window_.ring(compose.friends, 8, 'ring plain');
    const toAll = window_.ring(compose.public, 8, 'ring plain');
    const reach = window_.ring(compose.reach, 12);
    const top = { x: 300, y: 90, w: 1380 };
    const down = { x: 300, y: 2350, w: 1380 };
    const atRule = { x: 330, y: 330, w: 1320 };
    const writing = { x: 340, y: 100, w: 1300 };
    const closer = { x: 380, y: 190, w: 1220 };
    const everyone = [['אורי', 0], ['נועה', 1], ['יואב', 2], ['תמר', 3], ['רון', 4], ['מאיה', 5], ['גיל', 0], ['יעל', 1]].map(([text, color]) => {
      const node = face(text, color, 104);
      node.append(h('span', 'ok', svg('0 0 24 24', '', `<path d="${ICONS.check}"/>`)));
      return node;
    });
    const all = h('div', 'panel', h('h4', '', 'הפוסט מגיע לכל שמונת החברים'), h('div', 'faces', ...everyone));
    all.style.cssText = 'top:650px;left:440px;width:1040px';
    scene.append(all);

    return (t) => {
      const back = word(0, 'זה כל') - 0.75;
      const write = line(1).start - 0.35;
      window_.view(travel(t, [[0, top], [word(0, 'כל מה') - 0.1, top], [back - 0.15, down], [back + 0.75, atRule],
        [write, atRule], [write + 0.8, writing], [line(2).start - 0.2, writing], [line(2).start + 0.7, closer]]));
      window_.show('feed', 1);
      const typed = t >= write + 0.55 ? 1 : 0;
      // A press on the page changes it at once, so these two do not fade.
      const open = t >= word(1, 'או כל') - 0.1 && t < line(2).start - 0.25 ? 1 : 0;
      window_.show('compose', typed);
      window_.show('compose_public', open);
      rule.style.opacity = during(t, back + 0.7, write, 0.25);
      audience.style.opacity = during(t, word(1, 'בוחרים') - 0.1, word(1, 'רק') - 0.1, 0.2);
      toFriends.style.opacity = during(t, word(1, 'רק') - 0.1, word(1, 'או כל') - 0.1, 0.2);
      toAll.style.opacity = during(t, word(1, 'או כל') - 0.1, line(2).start - 0.3, 0.2);
      reach.style.opacity = during(t, word(2, 'כתוב') - 0.2, Infinity, 0.25);
      rise(all, t, word(2, 'הוא יגיע') - 0.3, { by: 30 });
      everyone.forEach((node, i) => {
        pop(node, t, word(2, 'הוא יגיע') - 0.1 + i * 0.09);
        pop(node.querySelector('.ok'), t, word(2, 'לכולם') - 0.1 + i * 0.07);
      });
    };
  },

  // Comments: all of them, in order. A link is a link. Pictures and videos.
  comments(scene, { line, word }) {
    const window_ = browser(scene, ['post']);
    const feedWindow = browser(scene, ['feed']);
    window_.show('post');
    feedWindow.show('feed');
    const { post, feed } = marks;
    const numbers = post.comments.map(([x, y], i) => {
      const badge = h('span', 'order', String(i + 1));
      return window_.put(badge, 320, y + 60);
    });
    const link = window_.ring(post.link, 16);
    const pointer = cursor(scene);
    const whole = { x: 300, y: 150, w: 1380 };
    const thread = { x: 240, y: 1520, w: 1420 };
    const atLink = { x: 420, y: 2330, w: 1160 };
    const photo = { x: 300, y: 2000, w: 1380 };
    const film = { x: 300, y: feed.video[1] - 330, w: 1380 };

    return (t) => {
      const go = word(0, 'כולן') - 0.5;
      window_.view(travel(t, [[0, whole], [go, whole], [go + 1.1, thread], [line(1).start - 0.4, thread], [line(1).start + 0.5, atLink]]));
      numbers.forEach((badge, i) => pop(badge, t, word(0, 'לפי') - 0.2 + i * 0.22, { until: line(1).start - 0.3 }));
      link.style.opacity = during(t, word(1, 'קישור') + 0.25, line(2).start - 0.3, 0.25);
      const [lx, ly] = window_.onStage(post.link[0] + 120, post.link[1] + 30);
      pointer.style.opacity = during(t, word(1, 'וכל') - 0.3, line(2).start - 0.3, 0.2);
      moveCursor(pointer, t, [[word(1, 'וכל') - 0.3, lx + 260, ly + 220], [word(1, 'ללחוץ') - 0.1, lx, ly]]);
      const swap = ramp(t, line(2).start - 0.3, 0.45);
      window_.node.style.opacity = 1 - swap;
      feedWindow.node.style.opacity = swap;
      feedWindow.view(travel(t, [[line(2).start, photo], [word(2, 'וסרטונים') - 0.5, photo], [word(2, 'וסרטונים') + 0.6, film]]));
    };
  },

  // Friends, the globe, and the update.
  around(scene, { line, word }) {
    const people = browser(scene, ['friends0', 'friends']);
    const home = browser(scene, ['home']);
    home.show('home');
    const { friends, home: spots } = marks;
    const search = people.ring(friends.search, 10);
    const ask = people.ring(friends.ask, 14);
    const chose = h('div', 'note', 'רק מי שבחר לשתף את המקום שלו');
    chose.style.cssText = 'left:330px;top:690px';
    scene.append(chose);
    const page = { x: 300, y: 80, w: 1380 };
    const all = { x: 0, y: 0, w: 2720 };
    const globe = { x: 1420, y: 110, w: 1300 };
    const update = { x: 0, y: 120, w: 1040 };
    const events = { x: 0, y: 880, w: 1040 };

    return (t) => {
      people.view(page);
      people.show('friends0', 1);
      people.show('friends', ramp(t, word(0, 'מחפשים') - 0.1, 0.35));
      search.style.opacity = during(t, word(0, 'מחפשים') - 0.1, word(0, 'ושולחים') - 0.15, 0.2);
      ask.style.opacity = during(t, word(0, 'ושולחים') - 0.15, line(1).start - 0.45, 0.2);
      const swap = ramp(t, line(1).start - 0.45, 0.45);
      people.node.style.opacity = 1 - swap;
      home.node.style.opacity = swap;
      const next = line(2).start - 0.5;
      home.view(travel(t, [[line(1).start, all], [line(1).start + 0.5, all], [line(1).start + 1.5, globe], [next, globe],
        [next + 1.1, update], [word(2, 'בקהילה') - 0.5, update], [word(2, 'בקהילה') + 0.9, events]]));
      rise(chose, t, word(1, 'רק') - 0.15, { until: next, by: 20 });
    };
  },

  // Small, just starting, and ours.
  closing(scene, { line, word }) {
    corners(scene);
    const phone = h('div', 'phone', image('shots/phone.png'));
    const now = h('div', 'small-now', 'הרשת עוד קטנה,', h('br'), 'ואנחנו רק מתחילים.', h('small', '', 'חסרים בה דברים, ונשמח לשמוע מכם מה.'));
    const ours = h('div', 'ours center', 'אבל היא שלנו.');
    const nobody = h('div', 'ours-note center', 'אף אחד לא מרוויח מהזמן שלכם כאן.');
    const logo = image('../public/favicon.svg', 'logo');
    logo.style.top = '250px';
    const welcome = h('div', 'welcome-end center', 'ברוכים הבאים לרשת החופשית');
    scene.append(phone, now, ours, nobody, logo, welcome);
    return (t) => {
      const turn = line(1).start - 0.35;
      phone.style.opacity = during(t, 0.2, turn, 0.4);
      phone.style.transform = `translateY(${60 * (1 - settle(ramp(t, 0.2, 0.9)))}px) rotate(${-3 + 3 * settle(ramp(t, 0.2, 0.9))}deg)`;
      rise(now, t, 0.3, { until: turn });
      rise(now.querySelector('small'), t, word(0, 'חסרים') - 0.15, { until: turn, by: 20 });
      const last = line(2).start - 0.4;
      pop(ours, t, line(1).start - 0.05, { until: last, from: 0.8 });
      rise(nobody, t, word(1, 'אף') - 0.15, { until: last, by: 24 });
      pop(logo, t, last + 0.25, { from: 0.3, length: 0.7 });
      rise(welcome, t, line(2).start - 0.1);
    };
  },
};

// The timeline: scenes one after the other, each as long as its narration plus a breath.
const caption = h('div', '', h('span'));
caption.id = 'caption';
let clock = 0;
const scenes = script.scenes.map((entry) => {
  const spoken = timing[entry.id];
  const node = h('section', 'scene');
  stage.append(node);
  const line = (i) => ({ start: LEAD + spoken.lines[i][0], end: LEAD + spoken.lines[i][1] });
  // When a word of a line is said, judged by how far into the line's letters it stands.
  const word = (i, text) => {
    const at = entry.lines[i].indexOf(text);
    if (at < 0) throw new Error(`"${text}" is not in line ${i} of ${entry.id}`);
    const letters = (part) => part.replace(/[\s.,:]/g, '').length;
    const { start, end } = line(i);
    return start + (end - start) * letters(entry.lines[i].slice(0, at)) / letters(entry.lines[i]);
  };
  const scene = { ...entry, node, start: clock, length: LEAD + spoken.length + TAIL, spoken, draw: SCENES[entry.id](node, { line, word }) };
  clock += scene.length;
  return scene;
});
stage.append(caption);
const duration = clock;

function seek(time) {
  const t = Math.max(0, Math.min(duration - 0.001, time));
  let said = null;
  scenes.forEach((scene, i) => {
    const local = t - scene.start;
    // A scene fades in over the end of the one before it, and the last one fades to nothing.
    const fadeIn = i === 0 ? ramp(local, 0, 0.6) : ramp(local, -FADE / 2, FADE);
    const fadeOut = i === scenes.length - 1 ? 1 - ramp(local, scene.length - 0.5, 0.5) : local < scene.length + FADE / 2 ? 1 : 0;
    const visible = local > -FADE / 2 && local < scene.length + FADE / 2;
    scene.node.style.display = visible ? '' : 'none';
    if (!visible) return;
    scene.node.style.opacity = fadeIn * fadeOut;
    scene.node.style.zIndex = i + 1;
    scene.draw(Math.max(0, Math.min(scene.length, local)));
    scene.spoken.lines.forEach(([from, until], n) => {
      const shown = during(local, LEAD + from - 0.1, LEAD + until + 0.25, 0.15);
      if (shown > 0 && (!said || shown >= said.shown)) said = { text: scene.lines[n], shown };
    });
  });
  caption.style.opacity = said?.shown ?? 0;
  caption.style.zIndex = 100;
  if (said && caption.firstChild.textContent !== said.text) caption.firstChild.textContent = said.text;
}

await document.fonts.ready;
await Promise.all([...document.images].map((picture) => picture.decode().catch(() => {})));
seek(0);
window.film = {
  duration,
  seek,
  // Where each recording starts, for the sound track.
  sound: scenes.map((scene) => ({ id: scene.id, at: scene.start + LEAD })),
};

// Opened by a person: fit the window and play, with the narration and a bar to move along it.
if (!new URLSearchParams(location.search).has('still')) {
  const controls = document.getElementById('controls');
  controls.hidden = false;
  const fit = () => {
    const scale = Math.min(innerWidth / 1920, (innerHeight - 44) / 1080);
    stage.style.transform = `scale(${scale})`;
    stage.style.left = `${(innerWidth - 1920 * scale) / 2}px`;
  };
  addEventListener('resize', fit);
  fit();
  const voices = scenes.map((scene) => Object.assign(new Audio(`audio/${scene.id}.wav`), { preload: 'auto' }));
  const scrub = document.getElementById('scrub');
  const button = document.getElementById('play');
  let playing = false;
  let now = Number(new URLSearchParams(location.search).get('t') ?? 0);
  let last = null;
  const sound = () => voices.forEach((voice, i) => {
    const local = now - window.film.sound[i].at;
    const on = playing && local >= 0 && local < voice.duration;
    if (on && voice.paused) {
      voice.currentTime = local;
      voice.play().catch(() => {});
    } else if (!on && !voice.paused) voice.pause();
  });
  const frame = (stamp) => {
    if (playing) now = Math.min(duration, now + (stamp - (last ?? stamp)) / 1000);
    last = stamp;
    if (now >= duration) playing = false;
    seek(now);
    sound();
    scrub.value = now / duration * 1000;
    document.getElementById('clock').textContent = `${now.toFixed(1)} / ${duration.toFixed(1)}`;
    button.textContent = playing ? 'עצור' : 'נגן';
    requestAnimationFrame(frame);
  };
  button.addEventListener('click', () => {
    if (now >= duration) now = 0;
    playing = !playing;
  });
  scrub.addEventListener('input', () => {
    now = scrub.value / 1000 * duration;
    voices.forEach((voice) => voice.pause());
  });
  requestAnimationFrame(frame);
}
