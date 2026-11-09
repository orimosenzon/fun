// Ori and Carmit manage quizzes (חידונים) here: titled, ordered selections from one shared pool of riddles.
// Images are shrunk in the browser and kept inside the Firestore documents as data URLs, so no Storage
// bucket (and no billing) is needed. Routing is by hash: '' = all quizzes, '#pool', '#quiz/<id>'.
import { app, db, fs, ADMINS, loadRiddles, loadQuizzes, loadLetters, primaryAnswer, DEFAULT_COVER, LEVELS, levelOf,
  hasHeading, fillHeading, safeUrl } from './fb.js';

const au = await import('https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js');
const auth = au.getAuth(app);
auth.languageCode = 'he';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;
const MAX_B64 = 380_000; // two images must fit together in one 1 MiB document

let riddles = [];        // the pool, newest first
let byId = new Map();
let quizzes = [];
let letters = [];        // marketing letters, each tied to one quiz (quizId)
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
const playUrl = (q, letter) => new URL(`index.html?q=${q.id}${letter ? `&m=${letter.id}` : ''}`, location.href).href;

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
  setTab({ '#pool': 'pool', '#path': 'path', '#letters': 'letters' }[location.hash] || 'quizzes');
  renderHome();
}

function setTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  $('tab-quizzes').classList.toggle('hidden', tab !== 'quizzes');
  $('tab-pool').classList.toggle('hidden', tab !== 'pool');
  $('tab-path').classList.toggle('hidden', tab !== 'path');
  $('tab-letters').classList.toggle('hidden', tab !== 'letters');
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
  const level = LEVELS.find((L) => L.id === levelOf(q)).name;
  meta.appendChild(el('div', 'sub', [`רמה ${level}`, n ? plural(n, 'חידה אחת', 'חידות') : 'אין חידות, לא יוצג'].join(' · ')));
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
    const level = LEVELS.find((L) => L.id === levelOf(q)).name;
    cap.append(el('b', '', q.title || 'ללא שם'), el('span', '', [`רמה ${level}`, count, q.shuffle && 'בסדר אקראי'].filter(Boolean).join(' · ')));
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
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', () => deleteRiddle(r));
    actions.append(edit, del);
    tile.append(img, body, actions);
    grid.appendChild(tile);
  }
  if (!grid.children.length) grid.appendChild(el('p', 'help', term ? 'לא נמצאו חידות.' : 'המאגר ריק.'));
}
$('pool-search').addEventListener('input', renderPool);
$('pool-new').addEventListener('click', () => openRiddle(null));

async function deleteRiddle(r) {
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
  $('play-link').href = playUrl(q);
  clearTimeout(headingTimer);
  $('heading-text').value = (q.heading || {}).text || '';
  setSaveState('');
  renderHeading();
  renderLetters();
  renderOrder();
  renderLevel();
  renderQuiz();
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

let saveTimer = null;
function setSaveState(s) {
  const st = $('save-state');
  st.textContent = { saving: 'שומרים…', saved: 'נשמר ✓', error: 'השמירה נכשלה' }[s] || '';
  st.className = `save-state ${s}`;
}

async function saveQuiz(patch) {
  const q = current;
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
function renderLevel() {
  document.querySelector(`input[name="quiz-level"][value="${levelOf(current)}"]`).checked = true;
}
document.querySelectorAll('input[name="quiz-level"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ level: r.value });
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

$('share-btn').addEventListener('click', () => copyLink(playUrl(current)));

async function copyLink(url) {
  try {
    await navigator.clipboard.writeText(url);
    toast('הקישור הועתק. אפשר להדביק ולשלוח.');
  } catch {
    prompt('הקישור לחידון:', url);
  }
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
  sel.disabled = !seen.size;
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

// ---------- marketing letters ----------

function letterRow(letter, withQuiz) {
  const q = quizzes.find((x) => x.id === letter.quizId);
  const li = el('li', 'letter-row');
  if (letter.image) {
    const img = el('img');
    img.src = letter.image;
    img.alt = '';
    li.appendChild(img);
  }
  const meta = el('div', 'meta');
  meta.appendChild(el('b', '', letter.name || 'מכתב ללא שם'));
  const text = (letter.text || '').replace(/\s+/g, ' ').trim();
  meta.appendChild(el('div', 'sub', text.length > 120 ? `${text.slice(0, 120)}…` : text || '(בלי טקסט)'));
  const extra = [];
  if (withQuiz) extra.push(`חידון: ${q ? q.title || 'ללא שם' : 'נמחק'}`);
  if (safeUrl(letter.linkUrl)) extra.push(`כפתור: ${letter.linkText || 'לפרטים נוספים'} → ${letter.linkUrl}`);
  if (extra.length) meta.appendChild(el('div', 'sub mono', extra.join(' · ')));
  li.appendChild(meta);
  const actions = el('div', 'tile-actions');
  if (q) {
    const copy = el('button', 'primary-btn small', 'העתקת קישור');
    copy.addEventListener('click', () => copyLink(playUrl(q, letter)));
    const view = el('a', 'link-btn', 'תצוגה ↗');
    view.href = playUrl(q, letter);
    view.target = '_blank';
    view.rel = 'noopener';
    actions.append(copy, view);
  }
  if (withQuiz && q) {
    const open = el('a', 'link-btn', 'לחידון');
    open.href = `#quiz/${q.id}`;
    actions.appendChild(open);
  } else {
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openLetter(letter));
    actions.appendChild(edit);
  }
  const del = el('button', 'link-btn del', 'מחיקה');
  del.addEventListener('click', () => deleteLetter(letter));
  actions.appendChild(del);
  li.appendChild(actions);
  return li;
}

function renderLetters() {
  const mine = letters.filter((l) => l.quizId === current.id);
  $('letters-count').textContent = mine.length || '';
  const list = $('letters-list');
  list.innerHTML = '';
  mine.forEach((l) => list.appendChild(letterRow(l, false)));
}

function renderAllLetters() {
  $('all-letters-count').textContent = letters.length;
  const list = $('all-letters');
  list.innerHTML = '';
  letters.forEach((l) => list.appendChild(letterRow(l, true)));
  if (!letters.length) list.appendChild(el('p', 'help', 'עוד אין מכתבים. פותחים חידון ולוחצים "+ מכתב נלווה חדש".'));
}

async function deleteLetter(letter) {
  if (!confirm(`למחוק את המכתב "${letter.name || 'ללא שם'}"?\nמי שיפתח את הקישור שלו יגיע לחידון בלי המכתב.`)) return;
  try {
    await fs.deleteDoc(fs.doc(db, 'letters', letter.id));
  } catch (err) { return toast(`המחיקה נכשלה: ${err.message}`); }
  letters = letters.filter((l) => l !== letter);
  toast('המכתב נמחק');
  current ? renderLetters() : renderAllLetters();
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
  $('letter-form-title').textContent = letter ? 'עריכת מכתב נלווה' : `מכתב נלווה לחידון "${current.title}"`;
  $('letter-name').value = letter?.name || '';
  $('letter-text').value = letter?.text || '';
  $('letter-link-text').value = letter?.linkText || '';
  $('letter-link-url').value = letter?.linkUrl || '';
  setLetterImage(letter?.image || '');
  $('letter-save').textContent = letter ? 'שמירה' : 'שמירה והעתקת הקישור';
  letterDialog.showModal();
}
$('letter-new').addEventListener('click', () => openLetter(null));
$('letter-close').addEventListener('click', () => letterDialog.close());
$('letter-cancel').addEventListener('click', () => letterDialog.close());
$('letter-img-remove').addEventListener('click', () => setLetterImage(''));
letterDialog.addEventListener('click', (e) => { if (e.target === letterDialog) letterDialog.close(); });

$('letter-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  letterError('');
  const data = {
    quizId: current.id,
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
      toast('המכתב עודכן. הקישור נשאר אותו קישור.');
    } else {
      const ref = await fs.addDoc(fs.collection(db, 'letters'), { ...data, created: fs.serverTimestamp() });
      const letter = { id: ref.id, ...data };
      letters.unshift(letter);
      letterDialog.close();
      renderLetters();
      return copyLink(playUrl(current, letter));
    }
    letterDialog.close();
    renderLetters();
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
    li.draggable = true;
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
    const out = el('button', 'icon-btn', '✕');
    out.title = 'הוצאה מהחידון (החידה נשארת במאגר)';
    out.addEventListener('click', () => setIds(current.riddleIds.filter((id) => id !== r.id)));
    li.append(box, el('span', 'grip', '⠿'), el('span', 'num', String(i + 1).padStart(2, '0')), img, meta, edit, out);
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
  quizzes.filter((q) => q !== current).forEach((q) => sel.appendChild(new Option(q.title || 'ללא שם', q.id)));
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
