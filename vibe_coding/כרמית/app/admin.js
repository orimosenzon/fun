// Ori and Carmit manage quizzes (חידונים) here: titled, ordered selections from one shared pool of riddles.
// Images are shrunk in the browser and kept inside the Firestore documents as data URLs, so no Storage
// bucket (and no billing) is needed. Routing is by hash: '' = all quizzes, '#pool', '#quiz/<id>'.
import { app, db, fs, ADMINS, LEVELS, loadRiddles, loadQuizzes, loadLetters, loadRatings, loadVisits, markOwnDevice, dayKey, primaryAnswer, DEFAULT_COVER, levelOf, levelLabel,
  hasHeading, fillHeading, safeUrl } from './fb.js?v=202611201410';

const au = await import('https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js');
const auth = au.getAuth(app);
auth.languageCode = 'he';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;
const MAX_B64 = 380_000; // two images must fit together in one 1 MiB document

let riddles = [];        // the pool, newest first
let byId = new Map();
let quizzes = [];
let letters = [];        // letters to send along with a quiz, before it or after it (any quiz)
let current = null;      // the quiz open in the quiz view
let picked = new Set();  // riddle ids checked in the quiz list, to copy or move to another quiz
let lastPick = null;     // anchor for shift-click ranges

let editing = null;      // id of the riddle being edited in the dialog, or null for a new one
let addToQuiz = null;    // a new riddle saved from the quiz view joins this quiz
let image = '';          // data URLs of the dialog form
let answerImage = '';

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const plural = (n, one, many) => (n === 1 ? one : `${n} ${many}`);
const riddleLabel = (r) => (r.answerType === 'image' ? 'תשובה בתמונה' : primaryAnswer(r));
const quizRiddles = (q) => q.riddleIds.map((id) => byId.get(id)).filter(Boolean);
const quizzesWith = (rid) => quizzes.filter((q) => q.riddleIds.includes(rid));
// A locked quiz (quiz.locked, "קבוע") is one that was sent and must stay as it is: nothing in it can change, and
// neither can the riddles in it, wherever they are edited from. These are its fields that the player sees.
const LOCKED_FIELDS = ['title', 'cover', 'riddleIds', 'shuffle', 'level', 'heading'];
const lockedWith = (rid) => quizzesWith(rid).filter((q) => q.locked);
const quizNames = (qs) => qs.map((q) => `"${q.title || 'ללא שם'}"`).join(', ');
// The link to send: the quiz on its own, with the letters chosen in it right now (b = before, a = after).
function playUrl(q) {
  const { opening, closing } = chosen(q);
  const extra = (opening ? `&b=${opening.id}` : '') + (closing ? `&a=${closing.id}` : '');
  return new URL(`index.html?q=${q.id}${extra}`, location.href).href;
}

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 2200);
}

// ---------- sign-in ----------

$('login-btn').addEventListener('click', async () => {
  try {
    await au.signInWithPopup(auth, new au.GoogleAuthProvider());
  } catch (e) {
    if (e.code !== 'auth/popup-closed-by-user') $('login-msg').textContent = `הכניסה נכשלה: ${e.message}`;
  }
});
$('logout-btn').addEventListener('click', () => au.signOut(auth));

au.onAuthStateChanged(auth, async (user) => {
  const ok = user && ADMINS.includes((user.email || '').toLowerCase());
  $('login-screen').classList.toggle('hidden', !!ok);
  $('who-box').classList.toggle('hidden', !ok);
  if (!ok) document.querySelectorAll('.view').forEach((v) => v.classList.add('hidden'));
  if (user && !ok) {
    $('login-msg').textContent = `החשבון ${user.email} לא מורשה לנהל חידות. רק אורי וכרמית יכולים.`;
    $('login-btn').textContent = 'כניסה עם חשבון אחר';
    au.signOut(auth);
  }
  if (ok) {
    $('who').textContent = user.email;
    markOwnDevice();   // from now on visits from this browser count as ours, not as a player's
    await reload();
    route();
  }
});

async function reload() {
  [riddles, quizzes, letters] = await Promise.all([loadRiddles(), loadQuizzes(),
    loadLetters().catch((e) => { console.error(e); return []; })]);
  byId = new Map(riddles.map((r) => [r.id, r]));
}

// ---------- routing ----------

window.addEventListener('hashchange', route);

function route() {
  if (!auth.currentUser) return;
  hidePeek();
  const [, quizId] = location.hash.match(/^#quiz\/(.+)$/) || [];
  const quiz = quizId && quizzes.find((q) => q.id === quizId);
  $('home-view').classList.toggle('hidden', !!quiz);
  $('quiz-view').classList.toggle('hidden', !quiz);
  if (quiz) return openQuiz(quiz);
  current = null;
  const tab = { '#pool': 'pool', '#path': 'path', '#letters': 'letters', '#stats': 'stats' }[location.hash] || 'quizzes';
  setTab(tab);
  renderHome();
  if (tab === 'stats') loadStats();
}

function setTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  $('tab-quizzes').classList.toggle('hidden', tab !== 'quizzes');
  $('tab-pool').classList.toggle('hidden', tab !== 'pool');
  $('tab-path').classList.toggle('hidden', tab !== 'path');
  $('tab-letters').classList.toggle('hidden', tab !== 'letters');
  $('tab-stats').classList.toggle('hidden', tab !== 'stats');
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  history.replaceState(null, '', t.dataset.tab === 'quizzes' ? location.pathname : `#${t.dataset.tab}`);
  route();
}));

// ---------- home: the quiz gallery ----------

function renderHome() {
  $('quiz-count').textContent = quizzes.length;
  $('pool-count').textContent = riddles.length;
  renderQuizGrid();
  renderPath();
  renderPool();
  renderAllLetters();
}

// ---------- statistics: how hard players found each riddle ----------

let ratings = null;      // every rating ever given; loaded when the tab is first opened
let visits = null;       // one per browser per day per page (the start screen or a quiz)

async function loadStats(force) {
  const body = $('stats-body');
  if (!ratings || force) {
    body.replaceChildren(el('p', 'help', 'טוענים…'));
    $('visits-body').replaceChildren();
    try {
      [ratings, visits] = await Promise.all([loadRatings(), loadVisits()]);
    } catch (e) {
      console.error(e);
      body.replaceChildren(el('p', 'help warn', 'לא הצלחנו לטעון את הנתונים. נסו לרענן.'));
      return;
    }
  }
  renderVisits();
  renderStats();
}
$('stats-refresh').addEventListener('click', () => loadStats(true));

// { easy, medium, hard, n, correct } over a list of ratings
function tally(list) {
  const t = { easy: 0, medium: 0, hard: 0, n: 0, correct: 0 };
  for (const r of list) {
    if (!(r.level in t)) continue;
    t[r.level]++;
    t.n++;
    if (r.correct) t.correct++;
  }
  return t;
}

// What the players think on the whole: the average of easy = 1, medium = 2, hard = 3, cut into thirds.
// With fewer than three ratings one person decides it, so no verdict yet.
const MIN_VERDICT = 3;
function verdict(t) {
  if (t.n < MIN_VERDICT) return null;
  const avg = (t.easy + 2 * t.medium + 3 * t.hard) / t.n;
  return LEVELS[avg < 5 / 3 ? 0 : avg <= 7 / 3 ? 1 : 2];
}

const pct = (part, n) => `${Math.round((100 * part) / n)}%`;

// One horizontal bar split into easy / medium / hard, each part as wide as its share.
function levelBar(t) {
  const bar = el('div', 'lv-bar');
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', LEVELS.map((L) => `${L.name}: ${t[L.id]}`).join(', '));
  for (const L of LEVELS) {
    if (!t[L.id]) continue;
    const seg = el('span', `lv-seg lv-${L.id}`, t[L.id] / t.n >= 0.1 ? String(t[L.id]) : '');
    seg.style.flexGrow = t[L.id];
    seg.title = `${L.name}: ${t[L.id]} מתוך ${t.n} (${pct(t[L.id], t.n)})`;
    bar.appendChild(seg);
  }
  return bar;
}

const countsText = (t) => LEVELS.map((L) => `${L.name} ${t[L.id]}`).join(' · ');

function statTile(value, label) {
  const tile = el('div', 'stat-tile');
  tile.append(el('b', '', value), el('span', '', label));
  return tile;
}

function riddleStatRow(r, t) {
  const li = el('li', 'stat-riddle');
  const img = el('img', 'thumb');
  img.src = r.image;
  img.alt = '';
  img.loading = 'lazy';
  const main = el('div', 'stat-main');
  const head = el('div', 'stat-head');
  head.append(el('b', '', riddleLabel(r)), el('span', 'help', `${plural(t.n, 'דירוג אחד', 'דירוגים')} · ${pct(t.correct, t.n)} פתרו`));
  main.append(head, levelBar(t), el('span', 'help stat-counts', countsText(t)));
  li.append(img, main);
  return li;
}

function quizStatCard(q, list) {
  const t = tally(list);
  const card = el('section', 'pane stat-quiz');
  const head = el('div', 'pane-head');
  const title = el('h2', '', q.title || 'חידון ללא שם');
  const meta = el('span', 'help', plural(t.n, 'דירוג אחד', 'דירוגים'));
  head.append(title, meta);
  card.append(head, levelBar(t), el('span', 'help stat-counts', countsText(t)));

  const v = verdict(t);
  const line = el('p', 'stat-verdict');
  const declared = levelLabel(q);
  if (!v) {
    line.textContent = `עוד מעט מדי דירוגים כדי לומר מה השחקנים חושבים (צריך לפחות ${MIN_VERDICT}).`;
    line.classList.add('help');
  } else {
    line.append('לפי השחקנים: ', el('b', '', `רמה ${v.name}`), declared ? ` · הוגדר: ${declared}` : ' · הוגדר: בלי רמה');
    if (declared && v.id !== levelOf(q)) line.append(el('span', 'stat-gap', ' ← שונה ממה שהוגדר'));
  }
  card.appendChild(line);

  // The riddles in the quiz's own order; a riddle since taken out of the quiz still shows its ratings, at the end.
  const byRiddle = Map.groupBy(list, (r) => r.riddle);
  const ids = [...q.riddleIds.filter((id) => byRiddle.has(id)), ...[...byRiddle.keys()].filter((id) => !q.riddleIds.includes(id))];
  const rows = ids.filter((id) => byId.has(id)).map((id) => riddleStatRow(byId.get(id), tally(byRiddle.get(id))));
  if (rows.length) {
    const details = el('details', 'stat-details');
    details.appendChild(el('summary', '', `לפי חידה (${rows.length})`));
    const ul = el('ul', 'stat-riddles');
    ul.append(...rows);
    details.appendChild(ul);
    card.appendChild(details);
  }
  return card;
}

const DAYS_SHOWN = 30;
const shortDay = (key) => `${Number(key.slice(8))}/${Number(key.slice(5, 7))}`;
const uniq = (list) => new Set(list.map((v) => v.visitor)).size;

// A bar for each of the last 30 days: how many different real visitors came that day.
function daysChart(real) {
  const perDay = Map.groupBy(real, (v) => v.day);
  const days = [];
  for (let i = DAYS_SHOWN - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key = dayKey(d);
    days.push({ key, n: uniq(perDay.get(key) || []) });
  }
  const max = Math.max(1, ...days.map((d) => d.n));
  const fig = el('figure', 'days-chart');
  fig.appendChild(el('figcaption', 'help', `מבקרים בכל יום, ${DAYS_SHOWN} הימים האחרונים (היום מימין). היום העמוס ביותר: ${max}.`));
  const plot = el('div', 'days-plot');
  plot.dir = 'ltr';
  days.forEach((d, i) => {
    const col = el('div', 'day-col');
    col.title = `${shortDay(d.key)}: ${plural(d.n, 'מבקר אחד', 'מבקרים')}`;
    const bar = el('i', 'day-bar');
    bar.style.height = `${(100 * d.n) / max}%`;
    if (!d.n) bar.classList.add('zero');
    // A date under every seventh bar, counted back from today.
    const label = el('span', 'day-label', (DAYS_SHOWN - 1 - i) % 7 === 0 ? shortDay(d.key) : '');
    const track = el('div', 'day-track');
    if (d.n === max && d.n > 0) track.appendChild(el('b', 'day-top', String(d.n)));
    track.appendChild(bar);
    col.append(track, label);
    plot.appendChild(col);
  });
  fig.appendChild(plot);
  return fig;
}

function renderVisits() {
  const body = $('visits-body');
  // A browser that was ever marked as ours is ours on all its days, also those before it first signed in.
  const ours = new Set(visits.filter((v) => v.own).map((v) => v.visitor));
  const real = visits.filter((v) => !ours.has(v.visitor));
  const own = visits.filter((v) => ours.has(v.visitor));
  if (!visits.length) {
    body.replaceChildren(el('p', 'help', 'עוד לא נרשמו כניסות.'));
    return;
  }
  const today = dayKey();
  const weekAgo = new Date();
  weekAgo.setDate(weekAgo.getDate() - 6);
  const weekKey = dayKey(weekAgo);
  const visitDays = (list) => new Set(list.map((v) => `${v.visitor} ${v.day}`)).size;
  const tiles = el('div', 'stat-tiles');
  tiles.append(
    statTile(String(uniq(real)), 'מבקרים אמיתיים בסך הכול'),
    statTile(String(uniq(real.filter((v) => v.day >= weekKey))), 'ב-7 הימים האחרונים'),
    statTile(String(uniq(real.filter((v) => v.day === today))), 'היום'),
    statTile(String(visitDays(real)), 'ביקורים (מבקר ביום)'),
  );
  const ownLine = el('p', 'help', own.length
    ? `בנפרד, הכניסות שלכם (כרמית ואורי): ${plural(visitDays(own), 'ביקור אחד', 'ביקורים')} מ-${plural(ours.size, 'דפדפן אחד', 'דפדפנים')}. הן לא נכללות באף מספר אחר כאן.`
    : 'עוד לא נרשמו כניסות מהמכשירים שלכם.');
  const out = [tiles, ownLine];
  if (real.length) {
    const chart = el('section', 'pane');
    chart.appendChild(daysChart(real));
    out.push(chart);
    // Per quiz: how many real visitors opened it, pressed start, and reached the results.
    const perQuiz = Map.groupBy(real.filter((v) => v.quiz), (v) => v.quiz);
    const rows = quizzes.filter((q) => perQuiz.has(q.id))
      .map((q) => {
        const list = perQuiz.get(q.id);
        return { q, opened: uniq(list), started: uniq(list.filter((v) => v.started)), finished: uniq(list.filter((v) => v.finished)),
          solo: uniq(list.filter((v) => v.solo)) };
      })
      .sort((a, b) => b.opened - a.opened);
    if (rows.length) {
      const pane = el('section', 'pane');
      pane.appendChild(el('h2', 'stat-h', 'לפי חידון'));
      const wrap = el('div', 'table-wrap');
      const table = el('table', 'visits-table');
      const head = el('tr');
      ['חידון', 'פתחו', 'התחילו', 'סיימו', 'הגיעו מקישור שנשלח'].forEach((h) => head.appendChild(el('th', '', h)));
      table.appendChild(head);
      for (const r of rows) {
        const tr = el('tr');
        tr.append(el('td', '', r.q.title || 'חידון ללא שם'), el('td', 'num', String(r.opened)), el('td', 'num', String(r.started)),
          el('td', 'num', String(r.finished)), el('td', 'num', String(r.solo)));
        table.appendChild(tr);
      }
      wrap.appendChild(table);
      pane.append(wrap, el('p', 'help', `דף הפתיחה (רשימת החידונים): ${plural(uniq(real.filter((v) => !v.quiz)), 'מבקר אחד', 'מבקרים')}.`));
      out.push(pane);
    }
  } else {
    out.push(el('p', 'help', 'עוד לא נרשמו מבקרים אמיתיים.'));
  }
  body.replaceChildren(...out);
}

function renderStats() {
  const body = $('stats-body');
  // Ratings of a quiz that was deleted since have nowhere to show.
  const byQuiz = Map.groupBy(ratings.filter((r) => quizzes.some((q) => q.id === r.quiz)), (r) => r.quiz);
  const all = tally([...byQuiz.values()].flat());
  $('stats-count').textContent = all.n || '';
  if (!all.n) {
    body.replaceChildren(el('p', 'help', 'עוד אין דירוגים. הם יופיעו כאן אחרי ששחקנים יסמנו כמה קשה הייתה להם חידה.'));
    return;
  }
  const tiles = el('div', 'stat-tiles');
  tiles.append(
    statTile(String(all.n), 'דירוגים בסך הכול'),
    ...LEVELS.map((L) => statTile(pct(all[L.id], all.n), `סימנו "${L.name}"`)),
  );
  const total = el('section', 'pane');
  total.append(el('h2', 'stat-h', 'כל החידונים יחד'), levelBar(all), el('span', 'help stat-counts', countsText(all)));
  // Quizzes with the most ratings first: that's where there is the most to learn.
  const cards = quizzes.filter((q) => byQuiz.has(q.id))
    .sort((a, b) => byQuiz.get(b.id).length - byQuiz.get(a.id).length)
    .map((q) => quizStatCard(q, byQuiz.get(q.id)));
  const silent = quizzes.filter((q) => !byQuiz.has(q.id) && q.riddleIds.length);
  body.replaceChildren(tiles, total, ...cards);
  if (silent.length) body.appendChild(el('p', 'help', `עוד בלי דירוגים: ${silent.map((q) => q.title || 'חידון ללא שם').join(', ')}.`));
}

// ---------- the path: which quizzes players see, and in what order ----------

const onPath = () => quizzes.filter((q) => Number.isInteger(q.path)).sort((a, b) => a.path - b.path);

// Writes the new positions: only quizzes whose position changed are saved.
async function setPath(ids) {
  const writes = [];
  for (const q of quizzes) {
    const i = ids.indexOf(q.id);
    const pos = i < 0 ? null : i;
    if ((Number.isInteger(q.path) ? q.path : null) === pos) continue;
    q.path = pos;
    writes.push(fs.updateDoc(fs.doc(db, 'quizzes', q.id), { path: pos, updated: fs.serverTimestamp() }));
  }
  renderPath();
  try {
    await Promise.all(writes);
    toast('הסדר נשמר');
  } catch (err) {
    toast(`השמירה נכשלה: ${err.message}`);
  }
}

function pathRow(q, i) {
  const li = el('li', 'q-item path-item');
  li.dataset.id = q.id;
  const img = coverImg(q);
  const meta = el('div', 'meta');
  meta.appendChild(el('b', '', q.title || 'ללא שם'));
  const n = quizRiddles(q).length;
  meta.appendChild(el('div', 'sub', [levelLabel(q), n ? plural(n, 'חידה אחת', 'חידות') : 'אין חידות, לא יוצג'].filter(Boolean).join(' · ')));
  const open = el('a', 'link-btn', 'עריכה');
  open.href = `#quiz/${q.id}`;
  if (i === null) {
    li.title = 'הוספה לסוף המסלול';
    li.append(img, meta, open, el('span', 'add', '+'));
    li.addEventListener('click', (e) => { if (e.target !== open) setPath([...onPath().map((x) => x.id), q.id]); });
  } else {
    const out = el('button', 'icon-btn', '✕');
    out.title = 'הוצאה מהמסלול (החידון לא נמחק)';
    out.addEventListener('click', () => setPath(onPath().map((x) => x.id).filter((id) => id !== q.id)));
    li.append(el('span', 'grip', '⠿'), el('span', 'num', String(i + 1).padStart(2, '0')), img, meta, open, out);
  }
  li.draggable = true;
  li.addEventListener('dragstart', (e) => {
    e.dataTransfer.setData('text/plain', `quiz:${q.id}`);
    e.dataTransfer.effectAllowed = 'move';
    requestAnimationFrame(() => li.classList.add('dragging'));
  });
  li.addEventListener('dragend', () => li.classList.remove('dragging'));
  return li;
}

function renderPath() {
  const path = onPath();
  const rest = quizzes.filter((q) => !path.includes(q));
  $('path-count').textContent = path.length;
  $('path-in-count').textContent = path.length;
  $('path-out-count').textContent = rest.length;
  $('path-fallback').classList.toggle('hidden', path.length > 0);
  const list = $('path-list');
  list.innerHTML = '';
  if (!path.length) list.appendChild(el('li', 'empty-drop', 'גוררים לכאן חידונים, או לוחצים עליהם'));
  path.forEach((q, i) => list.appendChild(pathRow(q, i)));
  const out = $('path-out');
  out.innerHTML = '';
  rest.forEach((q) => out.appendChild(pathRow(q, null)));
}

// Dropping on the path list inserts at the pointer: reorders a quiz on it, or adds one from the side.
{
  const list = $('path-list');
  const marker = el('li', 'drop-marker');
  const at = (y) => {
    const items = [...list.querySelectorAll('.q-item:not(.dragging)')];
    const i = items.findIndex((it) => y < it.getBoundingClientRect().top + it.offsetHeight / 2);
    return { items, i: i < 0 ? items.length : i };
  };
  list.addEventListener('dragover', (e) => {
    e.preventDefault();
    const { items, i } = at(e.clientY);
    if (items[i]) list.insertBefore(marker, items[i]);
    else list.appendChild(marker);
  });
  list.addEventListener('dragleave', (e) => { if (!list.contains(e.relatedTarget)) marker.remove(); });
  list.addEventListener('drop', (e) => {
    e.preventDefault();
    marker.remove();
    const [, id] = e.dataTransfer.getData('text/plain').match(/^quiz:(.+)$/) || [];
    if (!id || !quizzes.some((q) => q.id === id)) return;
    const { items, i } = at(e.clientY);
    const before = items[i]?.dataset.id;
    const ids = onPath().map((q) => q.id).filter((x) => x !== id);
    ids.splice(before ? ids.indexOf(before) : ids.length, 0, id);
    setPath(ids);
  });
}

function coverImg(q) {
  const img = el('img', q.cover ? 'cover' : 'cover default');
  img.src = q.cover || DEFAULT_COVER;
  img.alt = '';
  return img;
}

function renderQuizGrid() {
  const grid = $('quiz-grid');
  grid.innerHTML = '';

  const add = el('button', 'quiz-card new-quiz');
  add.innerHTML = '<span class="plus">+</span><span>חידון חדש</span>';
  add.addEventListener('click', createQuiz);
  grid.appendChild(add);

  for (const q of quizzes) {
    const rs = quizRiddles(q);
    const card = el('a', 'quiz-card');
    card.href = `#quiz/${q.id}`;
    // A few of its riddles peek out from behind the cover like tilted pages.
    const stack = el('div', 'stack');
    rs.slice(0, 3).forEach((r, i) => {
      const leaf = el('img', `leaf leaf-${i}`);
      leaf.src = r.image;
      leaf.alt = '';
      stack.appendChild(leaf);
    });
    const front = el('div', 'front');
    front.appendChild(coverImg(q));
    stack.appendChild(front);
    card.appendChild(stack);
    const cap = el('div', 'caption');
    const count = rs.length ? plural(rs.length, 'חידה אחת', 'חידות') : 'עוד אין חידות';
    cap.append(el('b', '', `${q.locked ? '🔒 ' : ''}${q.title || 'ללא שם'}`), el('span', '', [levelLabel(q), count, q.shuffle && 'בסדר אקראי'].filter(Boolean).join(' · ')));
    card.appendChild(cap);
    card.addEventListener('mouseenter', () => schedulePeek(card, q));
    card.addEventListener('mouseleave', hidePeek);
    grid.appendChild(card);
  }
}

async function createQuiz() {
  const data = { title: 'חידון חדש', cover: '', riddleIds: [], level: 'easy', by: auth.currentUser.email };
  const ref = await fs.addDoc(fs.collection(db, 'quizzes'), { ...data, created: fs.serverTimestamp() });
  quizzes.unshift({ id: ref.id, ...data });
  location.hash = `#quiz/${ref.id}`;
  setTimeout(() => { $('quiz-title').focus(); $('quiz-title').select(); }, 50);
}

// ---------- hover peek ----------

const canHover = matchMedia('(hover: hover)').matches;

function schedulePeek(card, q) {
  if (!canHover) return;
  clearTimeout(schedulePeek.timer);
  schedulePeek.timer = setTimeout(() => showPeek(card, q), 280);
}

function showPeek(card, q) {
  const rs = quizRiddles(q);
  const peek = $('peek');
  peek.innerHTML = '';
  peek.appendChild(el('div', 'peek-title', q.title || 'ללא שם'));
  if (!rs.length) {
    peek.appendChild(el('p', 'peek-empty', 'החידון ריק. לחיצה פותחת אותו כדי להוסיף חידות.'));
  } else {
    const grid = el('div', 'peek-grid');
    rs.slice(0, 12).forEach((r, i) => {
      const cell = el('div', 'peek-cell');
      const img = el('img');
      img.src = r.image;
      img.alt = '';
      cell.append(img, el('span', 'peek-ans', `${i + 1}. ${riddleLabel(r)}`));
      grid.appendChild(cell);
    });
    peek.appendChild(grid);
    if (rs.length > 12) peek.appendChild(el('p', 'peek-more', `ועוד ${rs.length - 12}`));
  }
  // Beside the card, on whichever side has room.
  const box = card.getBoundingClientRect();
  peek.classList.add('show');
  const w = peek.offsetWidth;
  const h = peek.offsetHeight;
  let x = box.left - w - 14;
  if (x < 8) x = box.right + 14;
  if (x + w > innerWidth - 8) x = Math.max(8, innerWidth - w - 8);
  const y = Math.min(Math.max(8, box.top), innerHeight - h - 8);
  peek.style.left = `${x}px`;
  peek.style.top = `${Math.max(8, y)}px`;
}

function hidePeek() {
  clearTimeout(schedulePeek.timer);
  $('peek').classList.remove('show');
}
addEventListener('scroll', hidePeek, { passive: true });

// ---------- pool tab ----------

const matches = (r, term) => !term || [r.question, r.hint, ...(r.answers || [])]
  .some((s) => (s || '').toLowerCase().includes(term));

function renderPool() {
  const term = $('pool-search').value.trim().toLowerCase();
  const grid = $('pool-grid');
  grid.innerHTML = '';
  for (const r of riddles.filter((x) => matches(x, term))) {
    const tile = el('div', 'pool-tile');
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const body = el('div', 'tile-body');
    body.appendChild(el('b', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)));
    if (r.hint) body.appendChild(el('div', 'sub', `רמז: ${r.hint}`));
    const qs = quizzesWith(r.id);
    const chips = el('div', 'chips');
    if (qs.length) {
      qs.forEach((q) => {
        const chip = el('a', 'chip', q.title || 'ללא שם');
        chip.href = `#quiz/${q.id}`;
        chips.appendChild(chip);
      });
    } else {
      chips.appendChild(el('span', 'chip lonely', 'לא בשום חידון'));
    }
    body.appendChild(chips);
    const actions = el('div', 'tile-actions');
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openRiddle(r));
    const clone = el('button', 'link-btn', 'שיבוט');
    clone.title = 'חידה חדשה במאגר, זהה לזו, שאפשר לשנות בלי לגעת במקור';
    clone.addEventListener('click', () => cloneRiddle(r));
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', () => deleteRiddle(r));
    actions.append(edit, clone, del);
    tile.append(img, body, actions);
    grid.appendChild(tile);
  }
  if (!grid.children.length) grid.appendChild(el('p', 'help', term ? 'לא נמצאו חידות.' : 'המאגר ריק.'));
}
$('pool-search').addEventListener('input', renderPool);
$('pool-new').addEventListener('click', () => openRiddle(null));

// A copy of a riddle as a new riddle in the pool, opened for editing. In a quiz the copy goes right after the
// original, or takes its place (replace), e.g. when the original can't be edited because another quiz is locked.
async function cloneRiddle(r, quiz = null, replace = false) {
  const { id, created, updated, ...data } = r;
  data.by = auth.currentUser.email;
  let copy;
  try {
    const ref = await fs.addDoc(fs.collection(db, 'riddles'), { ...data, created: fs.serverTimestamp() });
    copy = { id: ref.id, ...data };
    riddles.unshift(copy);
    byId.set(copy.id, copy);
    if (quiz && quiz === current && !quiz.locked) {
      const ids = [...quiz.riddleIds];
      const at = ids.indexOf(r.id);
      if (at < 0) ids.push(copy.id);
      else ids.splice(replace ? at : at + 1, replace ? 1 : 0, copy.id);
      await saveQuiz({ riddleIds: ids });
    }
  } catch (err) {
    return toast(`השיבוט נכשל: ${err.message}`);
  }
  current ? renderQuiz() : renderHome();
  toast(replace ? 'נוצר עותק והוא מחליף את המקור בחידון הזה' : 'נוצר עותק של החידה');
  openRiddle(copy);
  $('form-title').textContent = 'עריכת העותק';
}

async function deleteRiddle(r) {
  const locked = lockedWith(r.id);
  if (locked.length) return alert(`אי אפשר למחוק את החידה: היא נמצאת בחידון קבוע (${quizNames(locked)}).`);
  const qs = quizzesWith(r.id);
  const where = qs.length ? `\nהיא תוסר גם מ: ${qs.map((q) => q.title).join(', ')}.` : '';
  if (!confirm(`למחוק את החידה מהמאגר? אי אפשר לבטל את זה.${where}`)) return;
  await fs.deleteDoc(fs.doc(db, 'riddles', r.id));
  await Promise.all(qs.map((q) => {
    q.riddleIds = q.riddleIds.filter((id) => id !== r.id);
    return fs.updateDoc(fs.doc(db, 'quizzes', q.id), { riddleIds: fs.arrayRemove(r.id), updated: fs.serverTimestamp() });
  }));
  riddles = riddles.filter((x) => x.id !== r.id);
  byId.delete(r.id);
  toast('החידה נמחקה');
  current ? renderQuiz() : renderHome();
}

// ---------- one quiz ----------

function openQuiz(q) {
  current = q;
  picked = new Set();
  lastPick = null;
  $('quiz-title').value = q.title;
  $('side-search').value = '';
  $('side-source').value = '';
  clearTimeout(headingTimer);
  $('heading-text').value = (q.heading || {}).text || '';
  setSaveState('');
  renderHeading();
  renderLetters();
  renderOrder();
  renderLevel();
  renderAsk();
  renderQuiz();
  renderLock();
  scrollTo({ top: 0 });
}

function renderQuiz() {
  const q = current;
  const cover = $('cover-img');
  cover.src = q.cover || DEFAULT_COVER;
  cover.classList.toggle('default', !q.cover);
  $('cover-remove').classList.toggle('hidden', !q.cover);
  const rs = quizRiddles(q);
  $('in-count').textContent = rs.length;
  $('quiz-stats').textContent = rs.length ? plural(rs.length, 'חידה אחת', 'חידות') : 'עוד אין חידות';
  renderQuizList(rs);
  renderSideSource();
  renderSidePool();
}

// A locked quiz shows everything and lets nothing be changed; sending, playing, cloning and the letters stay open.
function renderLock() {
  const locked = !!current.locked;
  $('quiz-view').classList.toggle('locked', locked);
  $('lock-banner').classList.toggle('hidden', !locked);
  $('quiz-lock').textContent = locked ? '🔓 ביטול הקיבוע' : '🔒 קיבוע החידון';
  $('quiz-lock').title = locked ? 'החידון יחזור להיות ניתן לשינוי' : 'נועל את החידון ואת החידות שבו מפני שינויים, למשל אחרי שנשלח';
  for (const id of ['quiz-title', 'heading-text', 'heading-img-btn', 'heading-img-remove', 'heading-reset', 'cover-btn',
    'cover-remove', 'quiz-delete', 'quiz-new-riddle', 'pick-move']) $(id).disabled = locked;
  document.querySelectorAll('input[name="quiz-order"], input[name="quiz-level"]').forEach((r) => { r.disabled = locked; });
  if (locked) $('heading-from').disabled = true;
}

$('quiz-lock').addEventListener('click', async () => {
  const lock = !current.locked;
  if (!lock && !confirm(`לבטל את הקיבוע של "${current.title}"?
אחרי זה אפשר יהיה לשנות אותו ואת החידות שבו, ומי שכבר קיבל אותו יראה את השינויים.`)) return;
  await saveQuiz({ locked: lock });
  renderHeading();
  renderQuiz();
  renderLock();
  toast(lock ? 'החידון קבוע: אי אפשר לשנות אותו עד שמבטלים את הקיבוע' : 'הקיבוע בוטל');
});

// A new quiz with the same riddles (the very same ones, not copies) and the same settings. The copy is not
// locked and not on the players' path.
$('quiz-clone').addEventListener('click', async () => {
  const q = current;
  const data = { title: `${q.title || 'חידון'} (עותק)`, cover: q.cover || '', riddleIds: [...q.riddleIds], level: levelOf(q),
    shuffle: !!q.shuffle, heading: q.heading || { text: '', image: '' }, opening: q.opening || '', closing: q.closing || '',
    askRating: q.askRating !== false, by: auth.currentUser.email };
  try {
    const ref = await fs.addDoc(fs.collection(db, 'quizzes'), { ...data, created: fs.serverTimestamp() });
    quizzes.unshift({ id: ref.id, ...data });
    location.hash = `#quiz/${ref.id}`;
    toast(`נוצר עותק של "${q.title}". עכשיו עורכים את העותק.`);
  } catch (err) { toast(`השיבוט נכשל: ${err.message}`); }
});

let saveTimer = null;
function setSaveState(s) {
  const st = $('save-state');
  st.textContent = { saving: 'שומרים…', saved: 'נשמר ✓', error: 'השמירה נכשלה' }[s] || '';
  st.className = `save-state ${s}`;
}

async function saveQuiz(patch) {
  const q = current;
  // The last line of defence: whatever the screen let through, a locked quiz does not change.
  if (q.locked && LOCKED_FIELDS.some((k) => k in patch)) return toast('החידון קבוע, אי אפשר לשנות אותו. אפשר לשבט אותו או לבטל את הקיבוע.');
  Object.assign(q, patch);
  setSaveState('saving');
  try {
    await fs.updateDoc(fs.doc(db, 'quizzes', q.id), { ...patch, updated: fs.serverTimestamp() });
    if (q === current) setSaveState('saved');
  } catch (err) {
    setSaveState('error');
    toast(`השמירה נכשלה: ${err.message}`);
  }
}

$('quiz-title').addEventListener('input', () => {
  clearTimeout(saveTimer);
  setSaveState('saving');
  saveTimer = setTimeout(() => saveQuiz({ title: $('quiz-title').value.trim() }), 600);
});
$('quiz-title').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); });

// The player sees the riddles in the order set here, or reshuffled on every start.
function renderOrder() {
  const shuffle = !!current.shuffle;
  document.querySelector(`input[name="quiz-order"][value="${shuffle ? 'shuffle' : 'fixed'}"]`).checked = true;
  $('order-help').textContent = shuffle
    ? 'השחקן יקבל את החידות בסדר אקראי, חדש בכל משחק. הסדר כאן משמש רק לתצוגה המקדימה.'
    : 'גוררים כדי לשנות את הסדר. זה הסדר שבו השחקן יראה את החידות.';
}
document.querySelectorAll('input[name="quiz-order"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ shuffle: r.value === 'shuffle' });
  renderOrder();
}));

// The player climbs easy → medium → hard; a quiz from before levels existed counts as easy.
// 'none' is for a quiz that mixes levels: no label at all.
function renderLevel() {
  document.querySelector(`input[name="quiz-level"][value="${levelOf(current)}"]`).checked = true;
}
document.querySelectorAll('input[name="quiz-level"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ level: r.value });
}));

// quiz.askRating: whether the player is asked, after each riddle, how hard it was. On unless switched off.
function renderAsk() {
  document.querySelector(`input[name="quiz-ask"][value="${current.askRating === false ? 'no' : 'yes'}"]`).checked = true;
}
document.querySelectorAll('input[name="quiz-ask"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ askRating: r.value === 'yes' });
}));

$('cover-btn').addEventListener('click', () => $('cover-file').click());
$('cover-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    await saveQuiz({ cover: await readImage(file) });
    renderQuiz();
  } catch (err) { toast(err.message); }
});
$('cover-remove').addEventListener('click', async () => { await saveQuiz({ cover: '' }); renderQuiz(); });

// ---------- short links ----------
// What gets sent is a short link, orimosenzon.github.io/fun/carmit/?<code>. That page carries the preview
// (Wordplay & Wonder logo, "חידות כרמית כונסרי") and forwards to the quiz. links/<code> says where to: the quiz
// and its letters. The same quiz with the same letters always gets the same code back.
const SHORT_BASE = new URL('../../../carmit/', location.href).href;
const CODE_CHARS = 'abcdefghijkmnpqrstuvwxyz23456789';  // no l, o, 0, 1: nothing to misread

function randomCode() {
  return [...crypto.getRandomValues(new Uint8Array(6))].map((n) => CODE_CHARS[n % CODE_CHARS.length]).join('');
}

async function shortLink(q) {
  const { opening, closing } = chosen(q);
  const target = { q: q.id, b: opening ? opening.id : '', a: closing ? closing.id : '' };
  const key = `${target.q}|${target.b}|${target.a}`;
  const found = await fs.getDocs(fs.query(fs.collection(db, 'links'), fs.where('key', '==', key)));
  if (found.docs.length) return SHORT_BASE + '?' + found.docs[0].id;
  for (let i = 0; i < 5; i++) {
    const code = randomCode();
    if ((await fs.getDoc(fs.doc(db, 'links', code))).exists()) continue;
    await fs.setDoc(fs.doc(db, 'links', code), { ...target, key, by: auth.currentUser.email, created: fs.serverTimestamp() });
    return SHORT_BASE + '?' + code;
  }
  throw new Error('no free code');
}

$('share-btn').addEventListener('click', () => {
  const quiz = current;
  // Made inside the click itself: Safari on a phone only lets a click copy, not something that finishes later.
  const url = shortLink(quiz).catch((err) => {
    console.error(err);
    toast('לא הצלחנו ליצור קישור קצר, הועתק הקישור הארוך.');
    return playUrl(quiz);
  });
  copyLink(url);
});

// url may be a promise of the link (the short link is still being made).
async function copyLink(url) {
  try {
    if (typeof ClipboardItem === 'undefined') throw new Error('no ClipboardItem');
    const blob = Promise.resolve(url).then((u) => new Blob([u], { type: 'text/plain' }));
    await navigator.clipboard.write([new ClipboardItem({ 'text/plain': blob })]);
  } catch {
    const u = await url;
    try { await navigator.clipboard.writeText(u); } catch { return prompt('הקישור לחידון:', u); }
  }
  toast(`הקישור הועתק: ${await url}`);
}

// ---------- the heading above the quiz ----------

const headingKey = (h) => `${(h.text || '').trim()}|${h.image || ''}`;

function renderHeading() {
  const h = current.heading || {};
  $('heading-img').src = h.image || '';
  $('heading-img').classList.toggle('hidden', !h.image);
  $('heading-img-hint').classList.toggle('hidden', !!h.image);
  $('heading-img-remove').classList.toggle('hidden', !h.image);
  $('heading-reset').classList.toggle('hidden', !hasHeading(h));
  // The preview shows exactly what the player will see, the usual line when there is no heading.
  const preview = $('heading-preview');
  if (hasHeading(h)) fillHeading(preview, h);
  else preview.innerHTML = $('default-motif').innerHTML;
  // Headings of the other quizzes, each once, to take over as they are.
  const sel = $('heading-from');
  sel.innerHTML = '';
  sel.appendChild(new Option('לקחת כותרת מחידון אחר…', ''));
  const seen = new Map();
  for (const q of quizzes) {
    if (q === current || !hasHeading(q.heading) || headingKey(q.heading) === headingKey(h)) continue;
    const k = headingKey(q.heading);
    if (!seen.has(k)) seen.set(k, { heading: q.heading, titles: [] });
    seen.get(k).titles.push(q.title || 'ללא שם');
  }
  for (const [k, { heading, titles }] of seen) {
    const first = (heading.text || '').trim().split('\n')[0] || '(תמונה בלבד)';
    const label = `${heading.image ? '🖼 ' : ''}${first.length > 40 ? `${first.slice(0, 40)}…` : first}  ·  מתוך: ${titles.join(', ')}`;
    sel.appendChild(new Option(label, k));
  }
  if (seen.size) sel.appendChild(new Option('ההסבר הרגיל (עברית × English)', 'default'));
  sel.disabled = !seen.size || !!current.locked;
  if (!seen.size) sel.options[0].text = 'אין עוד כותרות בחידונים אחרים';
  sel.value = '';
}

function saveHeading(h) {
  saveQuiz({ heading: { text: (h.text || '').trim(), image: h.image || '' } });
  $('heading-text').value = current.heading.text;
  renderHeading();
}

let headingTimer = null;
$('heading-text').addEventListener('input', () => {
  clearTimeout(headingTimer);
  setSaveState('saving');
  // The preview follows every keystroke; the save waits for a pause.
  const h = { ...(current.heading || {}), text: $('heading-text').value };
  if (hasHeading(h)) fillHeading($('heading-preview'), h);
  else $('heading-preview').innerHTML = $('default-motif').innerHTML;
  headingTimer = setTimeout(async () => {
    await saveQuiz({ heading: { text: $('heading-text').value.trim(), image: (current.heading || {}).image || '' } });
    renderHeading();
  }, 600);
});
$('heading-from').addEventListener('change', (e) => {
  const k = e.target.value;
  if (!k) return;
  if (k === 'default') return saveHeading({ text: '', image: '' });
  const src = quizzes.find((q) => q !== current && hasHeading(q.heading) && headingKey(q.heading) === k);
  if (src) { saveHeading(src.heading); toast(`הכותרת של "${src.title}" הועתקה לחידון הזה`); }
});
$('heading-img-btn').addEventListener('click', () => $('heading-file').click());
$('heading-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    saveHeading({ text: $('heading-text').value, image: await readImage(file, 600, 120_000) });
  } catch (err) { toast(err.message); }
});
$('heading-img-remove').addEventListener('click', () => saveHeading({ text: $('heading-text').value, image: '' }));
$('heading-reset').addEventListener('click', () => {
  if (confirm('למחוק את הכותרת של החידון ולחזור להסבר הרגיל?')) saveHeading({ text: '', image: '' });
});

// ---------- letters ----------
// A letter stands on its own: text, a picture and a button, none of it tied to a quiz. In a quiz Carmit marks
// which letter goes before it (quiz.opening, רישא) and which after it (quiz.closing, סיפא), and the one share
// button turns that choice into the link. The link carries the letters itself, so a link already sent stays as it was.

function letterById(id) { return letters.find((l) => l.id === id); }
function letterGist(l) {
  const t = (l.text || '').replace(/\s+/g, ' ').trim();
  return t.length > 110 ? `${t.slice(0, 110)}…` : t || (l.image ? '(תמונה בלבד)' : '');
}
function letterName(l) { return l.name || 'מכתב ללא שם'; }
// The quiz's chosen letters, skipping one that was deleted since.
function chosen(q) { return { opening: letterById(q.opening) || null, closing: letterById(q.closing) || null }; }

function letterCard(letter) {
  const li = el('li', 'letter-row');
  li.dataset.id = letter.id;
  if (letter.image) {
    const img = el('img');
    img.src = letter.image;
    img.alt = '';
    li.appendChild(img);
  }
  const meta = el('div', 'meta');
  meta.appendChild(el('b', '', letterName(letter)));
  meta.appendChild(el('div', 'sub', letterGist(letter)));
  if (safeUrl(letter.linkUrl)) meta.appendChild(el('div', 'sub mono', `כפתור: ${letter.linkText || 'לפרטים נוספים'} → ${letter.linkUrl}`));
  li.appendChild(meta);
  return li;
}

// In the quiz: every letter, each marked as not sent, before the quiz or after it.
function renderLetters() {
  const list = $('letters-list');
  list.innerHTML = '';
  const { opening, closing } = chosen(current);
  for (const letter of letters) {
    const li = letterCard(letter);
    const role = letter === opening ? 'opening' : letter === closing ? 'closing' : 'none';
    li.classList.toggle('picked', role !== 'none');
    const seg = el('div', 'order-row letter-role');
    seg.setAttribute('role', 'radiogroup');
    seg.setAttribute('aria-label', `תפקיד המכתב "${letterName(letter)}"`);
    for (const [value, label] of [['none', 'לא נשלח'], ['opening', 'רישא: לפני'], ['closing', 'סיפא: אחרי']]) {
      const lab = el('label', 'seg');
      const input = el('input');
      input.type = 'radio';
      input.name = `role-${letter.id}`;
      input.value = value;
      input.checked = value === role;
      input.addEventListener('change', () => setRole(letter, value));
      lab.append(input, el('span', '', label));
      seg.appendChild(lab);
    }
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openLetter(letter));
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', () => deleteLetter(letter));
    const actions = el('div', 'tile-actions');
    actions.append(seg, edit, del);
    li.appendChild(actions);
    list.appendChild(li);
  }
  if (!letters.length) list.appendChild(el('p', 'help', 'עוד אין מכתבים. "+ מכתב חדש" כותב את הראשון.'));
  renderShare();
}

// One letter per place: marking a letter "before" takes the place from the one that held it.
function setRole(letter, role) {
  const { opening, closing } = chosen(current);
  const patch = { opening: opening ? opening.id : '', closing: closing ? closing.id : '' };
  if (patch.opening === letter.id) patch.opening = '';
  if (patch.closing === letter.id) patch.closing = '';
  if (role !== 'none') patch[role] = letter.id;
  saveQuiz(patch);
  renderLetters();
}

// The one place a link comes from: the share button, with the letters chosen right now.
function renderShare() {
  const { opening, closing } = chosen(current);
  $('play-link').href = playUrl(current);
  const parts = [opening && `לפני החידון: "${letterName(opening)}"`, closing && `אחרי החידון: "${letterName(closing)}"`].filter(Boolean);
  $('share-with').textContent = parts.length ? `הקישור יישלח עם מכתב ${parts.join(', ומכתב ')}.` : 'הקישור יישלח בלי מכתבים.';
}

// The letters tab: all letters, to write, edit and delete; which quizzes they are marked in.
function renderAllLetters() {
  $('all-letters-count').textContent = letters.length || '';
  const list = $('all-letters');
  list.innerHTML = '';
  for (const letter of letters) {
    const li = letterCard(letter);
    const uses = quizzes.filter((q) => q.opening === letter.id || q.closing === letter.id)
      .map((q) => `${q.title || 'ללא שם'} (${q.opening === letter.id ? 'לפני' : 'אחרי'})`);
    if (uses.length) li.querySelector('.meta').appendChild(el('div', 'sub', `מסומן ב: ${uses.join(', ')}`));
    const actions = el('div', 'tile-actions');
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openLetter(letter));
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', () => deleteLetter(letter));
    actions.append(edit, del);
    li.appendChild(actions);
    list.appendChild(li);
  }
  if (!letters.length) list.appendChild(el('p', 'help', 'עוד אין מכתבים.'));
}

function refreshLetters() {
  renderAllLetters();
  if (current) renderLetters();
}

async function deleteLetter(letter) {
  const uses = quizzes.filter((q) => q.opening === letter.id || q.closing === letter.id).map((q) => `"${q.title || 'ללא שם'}"`);
  const where = uses.length ? `\nהוא מסומן ב: ${uses.join(', ')}, והסימון יוסר.` : '';
  if (!confirm(`למחוק את המכתב "${letterName(letter)}"?${where}\nמי שיפתח קישור שנשלח איתו יגיע לחידון בלי המכתב.`)) return false;
  try {
    await fs.deleteDoc(fs.doc(db, 'letters', letter.id));
    // Quizzes that had it marked go back to no letter in that place.
    for (const q of quizzes) {
      const patch = {};
      if (q.opening === letter.id) patch.opening = '';
      if (q.closing === letter.id) patch.closing = '';
      if (Object.keys(patch).length) {
        Object.assign(q, patch);
        await fs.updateDoc(fs.doc(db, 'quizzes', q.id), patch);
      }
    }
  } catch (err) { return toast(`המחיקה נכשלה: ${err.message}`); }
  letters = letters.filter((l) => l !== letter);
  toast('המכתב נמחק');
  refreshLetters();
  return true;
}

const letterDialog = $('letter-dialog');
let letterEditing = null;
let letterImage = '';

function letterError(msg) {
  $('letter-error').textContent = msg;
  $('letter-error').classList.toggle('hidden', !msg);
}
function setLetterImage(data) {
  letterImage = data;
  showPreview($('letter-preview'), data);
  $('letter-img-remove').classList.toggle('hidden', !data);
}

function openLetter(letter) {
  letterEditing = letter;
  $('letter-form').reset();
  letterError('');
  $('letter-form-title').textContent = letter ? 'עריכת מכתב' : 'מכתב חדש';
  $('letter-name').value = letter?.name || '';
  $('letter-text').value = letter?.text || '';
  $('letter-link-text').value = letter?.linkText || '';
  $('letter-link-url').value = letter?.linkUrl || '';
  setLetterImage(letter?.image || '');
  $('letter-delete').classList.toggle('hidden', !letter);
  letterDialog.showModal();
}
$('letter-new').addEventListener('click', () => openLetter(null));
$('letter-new-home').addEventListener('click', () => openLetter(null));
$('letter-close').addEventListener('click', () => letterDialog.close());
$('letter-cancel').addEventListener('click', () => letterDialog.close());
$('letter-img-remove').addEventListener('click', () => setLetterImage(''));
$('letter-delete').addEventListener('click', async () => {
  if (await deleteLetter(letterEditing)) letterDialog.close();
});
letterDialog.addEventListener('click', (e) => { if (e.target === letterDialog) letterDialog.close(); });

$('letter-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  letterError('');
  const data = {
    name: $('letter-name').value.trim(),
    text: $('letter-text').value.trim(),
    image: letterImage,
    linkText: $('letter-link-text').value.trim(),
    linkUrl: $('letter-link-url').value.trim(),
    by: auth.currentUser.email,
  };
  if (!data.text && !data.image) return letterError('צריך לכתוב משהו במכתב, או לבחור תמונה.');
  if (data.linkUrl && !safeUrl(data.linkUrl)) return letterError('הכתובת של הכפתור לא נראית תקינה.');
  if (data.linkText && !data.linkUrl) return letterError('לכפתור יש טקסט אבל אין לאן הוא מוביל.');
  if (data.linkUrl) data.linkUrl = safeUrl(data.linkUrl);
  $('letter-save').disabled = true;
  try {
    if (letterEditing) {
      await fs.updateDoc(fs.doc(db, 'letters', letterEditing.id), { ...data, updated: fs.serverTimestamp() });
      Object.assign(letterEditing, data);
      toast('המכתב עודכן. גם מי שכבר קיבל קישור איתו יראה את הגרסה החדשה.');
    } else {
      const ref = await fs.addDoc(fs.collection(db, 'letters'), { ...data, created: fs.serverTimestamp() });
      letters.unshift({ id: ref.id, ...data });
      toast('המכתב נשמר');
    }
    letterDialog.close();
    refreshLetters();
  } catch (err) {
    letterError(`השמירה נכשלה: ${err.message}`);
  } finally {
    $('letter-save').disabled = false;
  }
});

$('quiz-delete').addEventListener('click', async () => {
  if (!confirm(`למחוק את החידון "${current.title}"?\nהחידות עצמן נשארות במאגר.`)) return;
  await fs.deleteDoc(fs.doc(db, 'quizzes', current.id));
  quizzes = quizzes.filter((q) => q !== current);
  toast('החידון נמחק');
  location.hash = '';
});

$('quiz-new-riddle').addEventListener('click', () => openRiddle(null, current));

function setIds(ids) {
  saveQuiz({ riddleIds: ids });
  renderQuiz();
}

function renderQuizList(rs) {
  const list = $('quiz-list');
  list.innerHTML = '';
  if (!rs.length) {
    list.appendChild(el('li', 'empty-drop', 'גוררים לכאן חידות מהמאגר, או לוחצים עליהן'));
  }
  rs.forEach((r, i) => {
    const li = el('li', picked.has(r.id) ? 'q-item picked' : 'q-item');
    li.draggable = !current.locked;
    li.dataset.id = r.id;
    const box = el('input', 'pick');
    box.type = 'checkbox';
    box.checked = picked.has(r.id);
    box.title = 'סימון להעתקה או העברה';
    box.addEventListener('click', (e) => togglePick(r.id, box.checked, e.shiftKey));
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const meta = el('div', 'meta');
    meta.appendChild(el('b', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)));
    if (r.hint || r.question) meta.appendChild(el('div', 'sub', [r.question, r.hint && `רמז: ${r.hint}`].filter(Boolean).join(' · ')));
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openRiddle(r));
    const clone = el('button', 'link-btn', 'שיבוט');
    clone.title = current.locked ? 'עותק של החידה במאגר, שאפשר לשנות' : 'עותק של החידה, מיד אחריה בחידון';
    clone.addEventListener('click', () => cloneRiddle(r, current));
    const out = el('button', 'icon-btn', '✕');
    out.title = 'הוצאה מהחידון (החידה נשארת במאגר)';
    out.addEventListener('click', () => setIds(current.riddleIds.filter((id) => id !== r.id)));
    if (current.locked) { edit.classList.add('hidden'); out.classList.add('hidden'); }
    li.append(box, el('span', 'grip', '⠿'), el('span', 'num', String(i + 1).padStart(2, '0')), img, meta, edit, clone, out);
    li.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', r.id);
      e.dataTransfer.effectAllowed = 'move';
      requestAnimationFrame(() => li.classList.add('dragging'));
    });
    li.addEventListener('dragend', () => li.classList.remove('dragging'));
    list.appendChild(li);
  });
  renderPickBar(rs);
}

// ---------- copy / move checked riddles to another quiz ----------

function togglePick(id, on, range) {
  const ids = quizRiddles(current).map((r) => r.id);
  const a = ids.indexOf(lastPick);
  const b = ids.indexOf(id);
  const span = range && a >= 0 ? ids.slice(Math.min(a, b), Math.max(a, b) + 1) : [id];
  span.forEach((x) => (on ? picked.add(x) : picked.delete(x)));
  lastPick = id;
  renderQuizList(quizRiddles(current));
}

const pickedIds = () => quizRiddles(current).map((r) => r.id).filter((id) => picked.has(id));

function renderPickBar(rs) {
  const n = pickedIds().length;
  $('pick-bar').classList.toggle('hidden', !rs.length);
  $('pick-all').checked = n > 0 && n === rs.length;
  $('pick-all').indeterminate = n > 0 && n < rs.length;
  $('pick-count').textContent = n ? `נבחרו ${n}` : 'בחירת הכל';
  $('pick-actions').classList.toggle('hidden', !n);
  const sel = $('pick-target');
  const keep = sel.value;
  sel.innerHTML = '';
  sel.appendChild(new Option('לחידון…', ''));
  // A locked quiz takes no new riddles.
  quizzes.filter((q) => q !== current && !q.locked).forEach((q) => sel.appendChild(new Option(q.title || 'ללא שם', q.id)));
  sel.appendChild(new Option('+ חידון חדש…', 'new'));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : '';
}

$('pick-all').addEventListener('change', (e) => {
  picked = new Set(e.target.checked ? current.riddleIds : []);
  lastPick = null;
  renderQuizList(quizRiddles(current));
});
$('pick-clear').addEventListener('click', () => { picked = new Set(); renderQuizList(quizRiddles(current)); });
$('pick-copy').addEventListener('click', () => transfer(false));
$('pick-move').addEventListener('click', () => transfer(true));

// Copy appends the checked riddles to the target quiz, skipping ones already there; move also takes them out of this one.
async function transfer(move) {
  const ids = pickedIds();
  const choice = $('pick-target').value;
  if (!ids.length) return;
  if (!choice) { $('pick-target').focus(); return toast('בוחרים קודם לאיזה חידון'); }
  if (move && current.locked) return toast('החידון קבוע, אפשר רק להעתיק ממנו.');
  let target;
  let dup = 0;
  try {
    if (choice === 'new') {
      const title = prompt('שם לחידון החדש:', 'חידון חדש');
      if (title == null) return;
      const data = { title: title.trim() || 'חידון חדש', cover: '', riddleIds: ids, level: 'easy', by: auth.currentUser.email };
      const ref = await fs.addDoc(fs.collection(db, 'quizzes'), { ...data, created: fs.serverTimestamp() });
      target = { id: ref.id, ...data };
      quizzes.unshift(target);
    } else {
      target = quizzes.find((q) => q.id === choice);
      const fresh = ids.filter((id) => !target.riddleIds.includes(id));
      dup = ids.length - fresh.length;
      const riddleIds = [...target.riddleIds, ...fresh];
      await fs.updateDoc(fs.doc(db, 'quizzes', target.id), { riddleIds, updated: fs.serverTimestamp() });
      target.riddleIds = riddleIds;
    }
    if (move) await saveQuiz({ riddleIds: current.riddleIds.filter((id) => !picked.has(id)) });
  } catch (err) {
    return toast(`הפעולה נכשלה: ${err.message}`);
  }
  const n = ids.length;
  const verb = move ? (n === 1 ? 'הועברה' : 'הועברו') : (n === 1 ? 'הועתקה' : 'הועתקו');
  const extra = !dup ? '' : dup === n ? ' (כבר היו שם)' : dup === 1 ? ' (אחת מהן כבר הייתה שם)' : ` (${dup} מהן כבר היו שם)`;
  toast(`${n === 1 ? 'חידה אחת' : `${n} חידות`} ${verb} ל"${target.title}"${extra}`);
  picked = new Set();
  renderQuiz();
  $('pick-target').value = target.id; // a quiz just created stays the target for the next batch
}

// The side pool can be narrowed to the riddles of one other quiz, to pull from it.
function renderSideSource() {
  const sel = $('side-source');
  const keep = sel.value;
  sel.innerHTML = '';
  sel.appendChild(new Option('מכל המאגר', ''));
  sel.appendChild(new Option('רק כאלה שלא בשום חידון', 'none'));
  quizzes.filter((q) => q !== current).forEach((q) => sel.appendChild(new Option(`מתוך: ${q.title || 'ללא שם'}`, q.id)));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : '';
}

let sideShown = [];

function renderSidePool() {
  const term = $('side-search').value.trim().toLowerCase();
  const source = $('side-source').value;
  const out = riddles.filter((r) => !current.riddleIds.includes(r.id));
  $('out-count').textContent = out.length;
  let from = out;
  if (source === 'none') from = out.filter((r) => !quizzesWith(r.id).length);
  else if (source) {
    const src = quizzes.find((q) => q.id === source);
    from = src ? quizRiddles(src).filter((r) => !current.riddleIds.includes(r.id)) : out;
  }
  sideShown = from.filter((x) => matches(x, term));
  const all = $('side-add-all');
  all.classList.toggle('hidden', !(source || term) || sideShown.length < 2);
  all.textContent = `+ הוספת כל ה-${sideShown.length} לסוף החידון`;
  const box = $('side-pool');
  box.innerHTML = '';
  for (const r of sideShown) {
    const item = el('button', 'side-item');
    item.draggable = true;
    item.title = 'הוספה לחידון';
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const n = quizzesWith(r.id).length;
    item.append(img, el('span', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)),
      el('span', 'uses', n ? `ב-${plural(n, 'חידון אחד', 'חידונים')}` : 'חדשה'), el('span', 'add', '+'));
    item.addEventListener('click', () => { setIds([...current.riddleIds, r.id]); toast('נוספה לסוף החידון'); });
    item.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', r.id);
      e.dataTransfer.effectAllowed = 'copyMove';
    });
    box.appendChild(item);
  }
  if (!box.children.length) {
    box.appendChild(el('p', 'help', term || source ? 'לא נמצאו חידות.' : 'כל החידות שבמאגר כבר בחידון הזה.'));
  }
}
$('side-search').addEventListener('input', renderSidePool);
$('side-source').addEventListener('change', renderSidePool);
$('side-add-all').addEventListener('click', () => {
  const n = sideShown.length;
  setIds([...current.riddleIds, ...sideShown.map((r) => r.id)]);
  toast(`${n} חידות נוספו לסוף החידון`);
});

// Dropping on the list inserts at the pointer: reorders a riddle already in it, or adds one from the pool.
const list = $('quiz-list');
const marker = el('li', 'drop-marker');

function dropIndex(y) {
  const items = [...list.querySelectorAll('.q-item:not(.dragging)')];
  const i = items.findIndex((it) => y < it.getBoundingClientRect().top + it.offsetHeight / 2);
  return { items, i: i < 0 ? items.length : i };
}

list.addEventListener('dragover', (e) => {
  e.preventDefault();
  const { items, i } = dropIndex(e.clientY);
  if (items[i]) list.insertBefore(marker, items[i]);
  else list.appendChild(marker);
});
list.addEventListener('dragleave', (e) => { if (!list.contains(e.relatedTarget)) marker.remove(); });
list.addEventListener('drop', (e) => {
  e.preventDefault();
  const id = e.dataTransfer.getData('text/plain');
  const { items, i } = dropIndex(e.clientY);
  marker.remove();
  if (!byId.has(id)) return;
  const before = items[i]?.dataset.id;
  const ids = current.riddleIds.filter((x) => x !== id);
  const at = before ? ids.indexOf(before) : ids.length;
  ids.splice(at, 0, id);
  setIds(ids);
});

// ---------- images ----------

function readImage(file, maxSide = MAX_SIDE, maxB64 = MAX_B64) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      let scale = Math.min(1, maxSide / Math.max(img.width, img.height));
      for (let attempt = 0; attempt < 8; attempt++) {
        const c = document.createElement('canvas');
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.drawImage(img, 0, 0, c.width, c.height);
        for (const q of [0.85, 0.75, 0.65, 0.55]) {
          const data = c.toDataURL('image/jpeg', q);
          if (data.length < maxB64) return resolve(data);
        }
        scale *= 0.8;
      }
      reject(new Error('התמונה גדולה מדי'));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('לא הצלחתי לקרוא את התמונה')); };
    img.src = url;
  });
}

function showPreview(img, data) {
  img.src = data || '';
  img.classList.toggle('hidden', !data);
  img.closest('.drop').classList.toggle('filled', !!data);
}

// A drop zone takes a click (file picker) or a dragged-in file.
function dropZone(zone, input, preview, set, onError = showError, size = []) {
  const take = async (file) => {
    if (!file) return;
    try { const data = await readImage(file, ...size); set(data); showPreview(preview, data); onError(''); }
    catch (err) { onError(err.message); }
  };
  input.addEventListener('change', (e) => { take(e.target.files[0]); e.target.value = ''; });
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('over');
    take(e.dataTransfer.files[0]);
  });
}
// A file dropped just outside a zone would otherwise make the browser leave the page to show it.
for (const ev of ['dragover', 'drop']) {
  addEventListener(ev, (e) => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); });
}
dropZone($('image-drop'), $('image-file'), $('image-preview'), (d) => { image = d; });
dropZone($('answer-drop'), $('answer-file'), $('answer-preview'), (d) => { answerImage = d; });
dropZone($('letter-drop'), $('letter-file'), $('letter-preview'), setLetterImage, letterError, [1000, 300_000]);

// ---------- answers ----------

const answerType = () => document.querySelector('input[name="answer-type"]:checked').value;

document.querySelectorAll('input[name="answer-type"]').forEach((r) => r.addEventListener('change', () => setType(answerType())));

function addAnswerRow(value = '', primary = false) {
  const row = document.createElement('div');
  row.className = 'answer-row';
  row.innerHTML = `
    <label class="star" title="התשובה העיקרית"><input type="radio" name="primary"><span>⭐</span></label>
    <input type="text" class="answer-text" autocomplete="off" placeholder="תשובה">
    <button type="button" class="link-btn remove" title="הסרה">✕</button>`;
  row.querySelector('.answer-text').value = value;
  row.querySelector('input[name="primary"]').checked = primary;
  row.querySelector('.remove').addEventListener('click', () => {
    const wasPrimary = row.querySelector('input[name="primary"]').checked;
    row.remove();
    const first = document.querySelector('#answers-list input[name="primary"]');
    if (wasPrimary && first) first.checked = true;
    if (!first) addAnswerRow('', true);
  });
  $('answers-list').appendChild(row);
  return row;
}
$('add-answer').addEventListener('click', () => {
  const noPrimary = !document.querySelector('#answers-list input[name="primary"]:checked');
  addAnswerRow('', noPrimary).querySelector('.answer-text').focus();
});

// ---------- the riddle dialog ----------

const dialog = $('riddle-dialog');

function showError(msg) {
  $('form-error').textContent = msg;
  $('form-error').classList.toggle('hidden', !msg);
}

function setType(type) {
  document.querySelector(`input[name="answer-type"][value="${type}"]`).checked = true;
  $('text-answers').classList.toggle('hidden', type !== 'text');
  $('image-answer').classList.toggle('hidden', type !== 'image');
}

function resetForm() {
  editing = null;
  image = '';
  answerImage = '';
  $('riddle-form').reset();
  showPreview($('image-preview'), '');
  showPreview($('answer-preview'), '');
  $('answers-list').innerHTML = '';
  addAnswerRow('', true);
  setType('text');
  showError('');
}

// r = null opens an empty form; quiz = the quiz a new riddle should join.
function openRiddle(r, quiz = null) {
  const locked = r ? lockedWith(r.id) : [];
  if (locked.length) {
    // From a quiz that is not locked itself, the editable copy takes the original's place there.
    const here = current && !current.locked && current.riddleIds.includes(r.id);
    const what = here ? `
בחידון "${current.title}" העותק יחליף את המקור. בחידון הקבוע דבר לא ישתנה.` : '';
    if (confirm(`החידה נמצאת בחידון קבוע (${quizNames(locked)}), ולכן אי אפשר לשנות אותה.
ליצור עותק שלה ולערוך אותו?${what}`)) cloneRiddle(r, here ? current : null, true);
    return;
  }
  resetForm();
  addToQuiz = quiz;
  $('form-title').textContent = r ? 'עריכת חידה' : quiz ? `חידה חדשה ב"${quiz.title}"` : 'חידה חדשה למאגר';
  if (r) {
    editing = r.id;
    image = r.image;
    answerImage = r.answerImage || '';
    showPreview($('image-preview'), image);
    showPreview($('answer-preview'), answerImage);
    $('question').value = r.question || '';
    $('hint').value = r.hint || '';
    setType(r.answerType);
    if (r.answerType === 'text' && r.answers.length) {
      $('answers-list').innerHTML = '';
      r.answers.forEach((a, i) => addAnswerRow(a, i === r.primary));
    }
    const n = quizzesWith(r.id).length;
    if (n > 1) $('form-title').textContent += ` (מופיעה ב-${n} חידונים)`;
  }
  dialog.showModal();
}

$('cancel-btn').addEventListener('click', () => dialog.close());
$('dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });

$('riddle-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  showError('');
  const type = answerType();
  const rows = [...document.querySelectorAll('#answers-list .answer-row')];
  let answers = [];
  let primary = 0;
  rows.forEach((row) => {
    const text = row.querySelector('.answer-text').value.trim();
    if (!text) return;
    if (row.querySelector('input[name="primary"]').checked) primary = answers.length;
    answers.push(text);
  });

  if (!image) return showError('צריך לבחור תמונה לחידה.');
  if (type === 'text' && !answers.length) return showError('צריך לכתוב לפחות תשובה אחת.');
  if (type === 'image' && !answerImage) return showError('צריך לבחור תמונה לתשובה.');
  if (type === 'image') { answers = []; primary = 0; }

  const data = {
    image,
    question: $('question').value.trim(),
    hint: $('hint').value.trim(),
    answerType: type,
    answers,
    primary,
    answerImage: type === 'image' ? answerImage : '',
    by: auth.currentUser.email,
  };
  $('save-btn').disabled = true;
  $('save-btn').textContent = 'שומרים…';
  try {
    if (editing) {
      await fs.updateDoc(fs.doc(db, 'riddles', editing), { ...data, updated: fs.serverTimestamp() });
      Object.assign(byId.get(editing), data);
      toast('החידה עודכנה');
    } else {
      const ref = await fs.addDoc(fs.collection(db, 'riddles'), { ...data, created: fs.serverTimestamp() });
      const r = { id: ref.id, ...data };
      riddles.unshift(r);
      byId.set(r.id, r);
      if (addToQuiz && addToQuiz === current) {
        await saveQuiz({ riddleIds: [...current.riddleIds, r.id] });
        toast('החידה נשמרה ונוספה לחידון');
      } else {
        toast('החידה נשמרה במאגר');
      }
    }
    dialog.close();
    current ? renderQuiz() : renderHome();
  } catch (err) {
    showError(`השמירה נכשלה: ${err.message}`);
  } finally {
    $('save-btn').disabled = false;
    $('save-btn').textContent = 'שמירה';
  }
});
