'use strict';
/* מסגרת: פרקים ותחנות, שמירת התקדמות, מספרים עשרוניים מדויקים, מילים בעברית, מנוע שאלות */

const Store = {
  key: 'ido-decimals-v1',
  data: {},
  load() { try { this.data = JSON.parse(localStorage.getItem(this.key)) || {}; } catch (e) { this.data = {}; } },
  save() { try { localStorage.setItem(this.key, JSON.stringify(this.data)); } catch (e) { /* בלי שמירה */ } },
  get(k, d) { return k in this.data ? this.data[k] : d; },
  set(k, v) { this.data[k] = v; this.save(); },
};

/* ---------- עזרים כלליים ---------- */
const ri = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const shuffle = arr => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ---------- מספרים עשרוניים מדויקים ----------
   כל מספר נשמר כמספר שלם של מיליוניות, כדי ש-0.1+0.2 ייצא 0.3 בדיוק. */
const U = 1e6;
const N = {
  // k חלקי 10 בחזקת p. למשל N.of(37, 2) הוא 0.37
  of(k, p = 0) { return Math.round(k * 10 ** (6 - p)); },
  parse(s) {
    s = String(s).trim().replace(/,/g, '.').replace(/\s+/g, '').replace(/[₪]/g, '');
    if (!/^-?(\d+\.?\d*|\.\d+)$/.test(s)) return null;
    const neg = s[0] === '-'; if (neg) s = s.slice(1);
    const [i, f = ''] = s.split('.');
    if (f.replace(/0+$/, '').length > 6) return null;
    const v = Number(i || '0') * U + Number((f + '000000').slice(0, 6));
    return neg ? -v : v;
  },
  str(v, minP = 0) {
    const neg = v < 0; v = Math.abs(Math.round(v));
    const i = Math.floor(v / U);
    const f = String(v % U).padStart(6, '0').replace(/0+$/, '').padEnd(minP, '0');
    return (neg ? '-' : '') + i + (f ? '.' + f : '');
  },
  // כמה ספרות אחרי הנקודה
  places(v) { const s = N.str(v); return s.includes('.') ? s.split('.')[1].length : 0; },
  mul(a, b) { return Math.round(a * b / U); },
  div(a, b) { return Math.round(a * U / b); },
  num(v) { return v / U; },
};

/* ---------- תצוגה ---------- */
// מתמטיקה בתוך טקסט עברי: תמיד משמאל לימין ומבודדת
const M = s => `<bdi dir="ltr" class="m">${s}</bdi>`;
const F = (n, d) => `<span class="frac"><span>${n}</span><span>${d}</span></span>`;
const PLACE_CLS = { 3: 'p3', 2: 'p2', 1: 'p1', 0: 'p0', '-1': 'pm1', '-2': 'pm2', '-3': 'pm3', '-4': 'pm4' };
const PLACE_NAME = { 3: 'אלפים', 2: 'מאות', 1: 'עשרות', 0: 'שלמים', '-1': 'עשיריות', '-2': 'מאיות', '-3': 'אלפיות' };
const PLACE_ONE = { 3: 'אלף', 2: 'מאה', 1: 'עשרת', 0: 'שלם', '-1': 'עשירית', '-2': 'מאית', '-3': 'אלפית' };
// ספרות צבועות לפי המקום שלהן. pad: כמה ספרות אחרי הנקודה להראות (אפסי רפאים)
function digitsHTML(str, opt = {}) {
  str = String(str);
  let [ip, fp = ''] = str.split('.');
  const hasDot = str.includes('.') || (opt.pad && opt.pad > 0);
  let out = '';
  for (let i = 0; i < ip.length; i++) out += `<span class="dg ${PLACE_CLS[ip.length - 1 - i] || ''}">${ip[i]}</span>`;
  if (hasDot) out += `<span class="dot">.</span>`;
  for (let i = 0; i < fp.length; i++) out += `<span class="dg ${PLACE_CLS[-(i + 1)] || ''}">${fp[i]}</span>`;
  for (let i = fp.length; i < (opt.pad || 0); i++) out += `<span class="dg ghost ${PLACE_CLS[-(i + 1)] || ''}">0</span>`;
  return `<bdi dir="ltr" class="digits">${out}</bdi>`;
}

/* ---------- מספרים במילים ---------- */
const HEB = {
  fu: ['', 'אחת', 'שתיים', 'שלוש', 'ארבע', 'חמש', 'שש', 'שבע', 'שמונה', 'תשע'],
  mu: ['', 'אחד', 'שניים', 'שלושה', 'ארבעה', 'חמישה', 'שישה', 'שבעה', 'שמונה', 'תשעה'],
  fteen: ['עשר', 'אחת עשרה', 'שתים עשרה', 'שלוש עשרה', 'ארבע עשרה', 'חמש עשרה', 'שש עשרה', 'שבע עשרה', 'שמונה עשרה', 'תשע עשרה'],
  mteen: ['עשרה', 'אחד עשר', 'שנים עשר', 'שלושה עשר', 'ארבעה עשר', 'חמישה עשר', 'שישה עשר', 'שבעה עשר', 'שמונה עשר', 'תשעה עשר'],
  tens: ['', '', 'עשרים', 'שלושים', 'ארבעים', 'חמישים', 'שישים', 'שבעים', 'שמונים', 'תשעים'],
  hund: ['', 'מאה', 'מאתיים', 'שלוש מאות', 'ארבע מאות', 'חמש מאות', 'שש מאות', 'שבע מאות', 'שמונה מאות', 'תשע מאות'],
  // n בין 1 ל-999, fem = נקבה
  num(n, fem) {
    const parts = [];
    const h = Math.floor(n / 100), r = n % 100, t = Math.floor(r / 10), u = r % 10;
    if (h) parts.push(this.hund[h]);
    if (r >= 10 && r < 20) parts.push((fem ? this.fteen : this.mteen)[r - 10]);
    else {
      if (t) parts.push(this.tens[t]);
      if (u) parts.push((fem ? this.fu : this.mu)[u]);
    }
    if (parts.length > 1) parts[parts.length - 1] = 'ו' + parts[parts.length - 1];
    return parts.join(' ');
  },
  // "שתי עשיריות", "עשירית אחת", "שלושים וחמש מאיות"
  count(n, one, many, fem = true) {
    if (n === 1) return `${one} ${fem ? 'אחת' : 'אחד'}`;
    if (n === 2) return `${fem ? 'שתי' : 'שני'} ${many}`;
    return `${this.num(n, fem)} ${many}`;
  },
  // קריאה של מספר עשרוני: "שני שלמים ושלושים וחמש מאיות"
  read(v) {
    const s = N.str(v); const [ip, fp = ''] = s.split('.');
    const W = Number(ip);
    const wholes = W ? this.count(W, 'שלם', 'שלמים', false) : '';
    let frac = '';
    if (fp) {
      const k = Number(fp), names = { 1: ['עשירית', 'עשיריות'], 2: ['מאית', 'מאיות'], 3: ['אלפית', 'אלפיות'] }[fp.length];
      frac = this.count(k, names[0], names[1], true);
    }
    if (wholes && frac) return wholes + ' ו' + frac;
    return wholes || frac || 'אפס';
  },
};

/* ---------- קונפטי ---------- */
function confetti(n = 70) {
  const box = document.createElement('div'); box.className = 'confetti';
  const cols = ['#2563eb', '#ea580c', '#7c3aed', '#16a34a', '#db2777', '#eab308'];
  for (let i = 0; i < n; i++) {
    const c = document.createElement('i');
    c.style.left = Math.random() * 100 + 'vw'; c.style.background = pick(cols);
    c.style.animationDelay = Math.random() * .5 + 's'; c.style.animationDuration = 1.4 + Math.random() * 1.2 + 's';
    box.appendChild(c);
  }
  document.body.appendChild(box); setTimeout(() => box.remove(), 3200);
}

/* ---------- SVG ---------- */
const S = {
  ns: 'http://www.w3.org/2000/svg',
  el(tag, attrs = {}, parent) {
    const e = document.createElementNS(this.ns, tag);
    for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  },
  text(parent, x, y, s, attrs = {}) { const t = this.el('text', { x, y, ...attrs }, parent); t.textContent = s; return t; },
  svg(w, h, cls = 'board') { return this.el('svg', { viewBox: `0 0 ${w} ${h}`, class: cls }); },
  // נקודת לחיצה בקואורדינטות של ה-SVG
  pt(svg, e) { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); },
  clear(g) { while (g.firstChild) g.removeChild(g.firstChild); },
};

/* ---------- מנוע שאלות ----------
   שאלה: { q, fig, type: 'choice'|'input'|'custom', options:[{h,v}], answer, hint, explain, unit, mount }
   בחירה: answer הוא ה-v הנכון. קלט: answer במיליוניות (או accept(str) משלך). */
function Quiz(el, opt) {
  const rounds = opt.rounds || 8;
  let i = 0, score = 0, streak = 0, tries = 0, q = null, finished = false, reported = false;
  el.classList.add('quiz');
  el.innerHTML = `
    <div class="qz-top"><span class="qz-prog"></span><span class="qz-dots"></span><span class="qz-streak"></span></div>
    <div class="qz-body">
      <div class="qz-q"></div><div class="qz-fig"></div><div class="qz-ans"></div><div class="qz-fb"></div>
      <div class="qz-next"></div>
    </div>`;
  const $ = s => el.querySelector(s);
  const results = [];

  function top() {
    $('.qz-prog').textContent = finished ? 'סיימת!' : `שאלה ${i + 1} מתוך ${rounds}`;
    $('.qz-dots').innerHTML = Array.from({ length: rounds }, (_, k) => `<i class="${results[k] === true ? 'ok' : results[k] === false ? 'no' : k === i && !finished ? 'cur' : ''}"></i>`).join('');
    $('.qz-streak').innerHTML = streak >= 2 ? `🔥 ${streak} ברצף` : '';
  }

  function ask() {
    q = opt.gen(i); tries = 0; el._q = q; // _q: לבדיקות אוטומטיות
    $('.qz-q').innerHTML = q.q;
    const fig = $('.qz-fig'); fig.innerHTML = '';
    if (typeof q.fig === 'function') q.fig(fig); else if (q.fig) fig.innerHTML = q.fig;
    $('.qz-fb').innerHTML = ''; $('.qz-fb').className = 'qz-fb';
    $('.qz-next').innerHTML = '';
    const ans = $('.qz-ans'); ans.innerHTML = '';
    if (q.type === 'choice') {
      ans.innerHTML = `<div class="opts">${q.options.map((o, k) => `<button class="opt ${q.optCls || ''}" data-k="${k}">${o.h}</button>`).join('')}</div>`;
      ans.querySelectorAll('.opt').forEach(b => b.onclick = () => {
        const o = q.options[+b.dataset.k];
        const ok = q.check ? q.check(o.v) : o.v === q.answer;
        if (ok) { b.classList.add('right'); ans.querySelectorAll('.opt').forEach(x => x.disabled = true); }
        else { b.classList.add('wrong'); b.disabled = true; }
        judge(ok, () => {
          ans.querySelectorAll('.opt').forEach(x => { x.disabled = true; const oo = q.options[+x.dataset.k]; if (q.check ? q.check(oo.v) : oo.v === q.answer) x.classList.add('right'); });
        }, q.options.length <= 2 ? 1 : 2);
      });
    } else if (q.type === 'input') {
      ans.innerHTML = `<div class="inrow"><input class="num-in" type="text" inputmode="decimal" dir="ltr" autocomplete="off" aria-label="תשובה">${q.unit ? `<span class="unit">${q.unit}</span>` : ''}<button class="btn primary">בדיקה</button></div>`;
      const inp = ans.querySelector('input'), btn = ans.querySelector('button');
      const go = () => {
        const s = inp.value; if (!s.trim()) return;
        const v = N.parse(s);
        if (!q.accept && v === null) { fb('bad', 'זה לא נראה כמו מספר. אפשר לכתוב למשל ' + M('0.25') + '.'); return; }
        const ok = q.accept ? q.accept(s, v) : v === q.answer;
        inp.classList.toggle('wrong', !ok); inp.classList.toggle('right', ok);
        q.onTry && q.onTry(v, ok);
        judge(ok, () => { inp.value = q.show || N.str(q.answer); inp.classList.remove('wrong'); inp.classList.add('right'); inp.disabled = true; btn.disabled = true; });
        if (ok) { inp.disabled = true; btn.disabled = true; }
      };
      btn.onclick = go; inp.onkeydown = e => { if (e.key === 'Enter') go(); };
      setTimeout(() => { if (el.offsetParent) inp.focus({ preventScroll: true }); }, 30);
    } else if (q.type === 'custom') {
      q.mount(ans, ok => judge(ok, q.reveal || (() => {}), q.maxTries || 2));
    }
    top();
  }

  function fb(cls, html) { const f = $('.qz-fb'); f.className = 'qz-fb ' + cls; f.innerHTML = html; }

  // ok: נכון או לא. reveal: מראה את התשובה הנכונה. maxT: אחרי כמה טעויות מגלים
  function judge(ok, reveal, maxT = 2) {
    if (finished) return;
    tries++;
    if (ok) {
      const first = tries === 1;
      if (first) { score++; streak++; } else streak = 0;
      results[i] = first;
      fb('ok', `<b>${pick(first ? ['נכון! ✓', 'בדיוק! ✓', 'יפה מאוד! ✓', 'מצוין! ✓'] : ['נכון, הפעם זה יצא ✓', 'יפה, תיקנת ✓'])}</b>${q.explain ? '<div class="why">' + q.explain + '</div>' : ''}`);
      nextBtn();
    } else if (tries < maxT) {
      fb('bad', `<b>עוד לא.</b> ${q.hint || 'נסה שוב.'}`);
    } else {
      streak = 0; results[i] = false;
      reveal();
      const s = q.show || (q.type === 'input' ? N.str(q.answer) : '');
      fb('bad', `<b>${s ? 'התשובה היא ' + M(s) : 'התשובה הנכונה מסומנת בירוק'}.</b>${q.explain ? '<div class="why">' + q.explain + '</div>' : ''}`);
      nextBtn();
    }
    top();
  }

  function nextBtn() {
    const last = i === rounds - 1;
    $('.qz-next').innerHTML = `<button class="btn primary">${last ? 'לתוצאות' : 'לשאלה הבאה ←'}</button>`;
    const b = $('.qz-next button'); b.onclick = () => { if (last) end(); else { i++; ask(); } };
    setTimeout(() => b.focus({ preventScroll: true }), 30);
  }

  function end() {
    finished = true; top();
    const stars = score >= rounds - 1 ? '⭐⭐⭐' : score >= rounds * .6 ? '⭐⭐' : '⭐';
    if (score >= rounds - 1) confetti();
    $('.qz-body').innerHTML = `<div class="qz-end"><div class="stars">${stars}</div>
      <p><b>${score} מתוך ${rounds}</b> נכונות בניסיון הראשון.</p>
      <p class="muted">${score >= rounds - 1 ? 'שולט! אפשר להמשיך הלאה.' : score >= rounds * .6 ? 'טוב מאוד. אפשר להמשיך, או לעשות עוד סיבוב לחיזוק.' : 'שווה לעשות עוד סיבוב. כל טעות היא הזדמנות להבין משהו.'}</p>
      <button class="btn again">עוד סיבוב 🔁</button></div>`;
    $('.again').onclick = restart;
    if (!reported) { reported = true; opt.onDone && opt.onDone(score, rounds); }
  }

  function restart() {
    i = 0; score = 0; streak = 0; finished = false; results.length = 0;
    $('.qz-body').innerHTML = `<div class="qz-q"></div><div class="qz-fig"></div><div class="qz-ans"></div><div class="qz-fb"></div><div class="qz-next"></div>`;
    ask();
  }
  ask();
  return { restart };
}

// שאלת בחירה מהירה: בונה אפשרויות מערכים ומערבב
function choiceQ(q, right, wrongs, extra = {}) {
  const opts = shuffle([right, ...wrongs.filter(w => w !== right)].filter((x, k, a) => a.indexOf(x) === k));
  return { q, type: 'choice', options: opts.map(o => ({ h: extra.fmt ? extra.fmt(o) : o, v: o })), answer: right, ...extra };
}

/* ---------- משימות בתוך תחנה ----------
   רשימת משימות שמתגלות אחת אחרי השנייה. check() נקרא בכל שינוי. */
function Missions(el, list, onAll) {
  let k = 0;
  el.classList.add('missions');
  function draw() {
    el.innerHTML = list.map((m, j) => `<div class="mis ${j < k ? 'done' : j === k ? 'cur' : 'later'}">
      <span class="mi">${j < k ? '✓' : j === k ? '🎯' : '○'}</span><span>${j <= k ? m.t : 'משימה נוספת'}</span></div>`).join('')
      + (k >= list.length ? `<div class="mis all">🏆 כל המשימות הושלמו!</div>` : '');
  }
  draw();
  return {
    get index() { return k; },
    get current() { return list[k]; },
    check(state) {
      if (k >= list.length) return false;
      if (list[k].ok(state)) {
        const m = list[k]; k++; draw();
        m.after && m.after(state);
        if (k >= list.length) onAll && onAll();
        return true;
      }
      return false;
    },
  };
}

/* ---------- פרקים ותחנות ---------- */
const App = {
  chapters: [],
  add(ch) { this.chapters.push(ch); },
  cur: null,

  start() {
    Store.load();
    const main = document.querySelector('main');
    this.chapters.forEach((ch, n) => {
      const sec = document.createElement('section');
      sec.className = 'chapter'; sec.id = ch.id; sec.hidden = true;
      if (ch.steps) {
        sec.innerHTML = `<h2><span class="num">${n}</span>${ch.title || ch.short}</h2>
          ${ch.intro ? `<div class="intro">${ch.intro}</div>` : ''}
          <div class="stepper"></div><div class="stage"></div>
          <div class="stepnav"><button class="btn prev">→ הקודם</button><span class="where"></span><button class="btn primary next">הבא ←</button></div>`;
      } else {
        sec.innerHTML = `<div class="stage single"></div>`;
      }
      main.appendChild(sec);
      ch.sec = sec; ch.rendered = {}; ch.at = Store.get('at', {})[ch.id] || 0;
      if (ch.steps) {
        sec.querySelector('.prev').onclick = () => this.step(ch, ch.at - 1);
        sec.querySelector('.next').onclick = () => {
          if (ch.at < ch.steps.length - 1) this.step(ch, ch.at + 1);
          else { const nx = this.chapters[n + 1]; if (nx) this.go(nx.id); }
        };
      }
    });
    this.renderNav();
    document.addEventListener('click', e => {
      const b = e.target.closest('[data-go]');
      if (b) { e.preventDefault(); this.go(b.dataset.go); }
    });
    window.addEventListener('hashchange', () => { const id = location.hash.slice(1); if (id !== this.cur && this.byId(id)) this.go(id); });
    const h = location.hash.slice(1);
    this.go(this.byId(h) ? h : Store.get('last', 'ch0'));
  },
  byId(id) { return this.chapters.find(c => c.id === id); },

  go(id) {
    const ch = this.byId(id); if (!ch) return;
    this.cur = id;
    this.chapters.forEach(c => { c.sec.hidden = c.id !== id; });
    document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b.dataset.go === id));
    const act = document.querySelector('#nav button.active');
    if (act) act.scrollIntoView({ block: 'nearest', inline: 'center' });
    if (ch.steps) this.step(ch, clamp(ch.at, 0, ch.steps.length - 1), true);
    else if (!ch.rendered.main) { ch.rendered.main = true; ch.render(ch.sec.querySelector('.stage')); }
    else if (ch.onShow) ch.onShow();
    Store.set('last', id);
    if (location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
    window.scrollTo({ top: 0 });
  },

  stepsDone(ch) { return Store.get('sd', {})[ch.id] || []; },
  isDone(id) { return !!Store.get('done', {})[id]; },

  step(ch, k, noScroll) {
    k = clamp(k, 0, ch.steps.length - 1);
    ch.at = k;
    const at = Store.get('at', {}); at[ch.id] = k; Store.set('at', at);
    const stage = ch.sec.querySelector('.stage');
    if (!ch.rendered[k]) {
      const d = document.createElement('div'); d.className = 'step'; d.dataset.k = k; d.id = `${ch.id}-s${k}`;
      const st = ch.steps[k];
      d.innerHTML = `<h3 class="st-title">${st.t}</h3><div class="st-body"></div>`;
      stage.appendChild(d); ch.rendered[k] = d;
      st.r(d.querySelector('.st-body'), () => this.stepDone(ch, k));
      if (st.auto) this.stepDone(ch, k);
    }
    stage.querySelectorAll('.step').forEach(s => { s.hidden = +s.dataset.k !== k; });
    this.renderStepper(ch);
    ch.steps[k].onShow && ch.steps[k].onShow();
    if (!noScroll) ch.sec.querySelector('.stepper').scrollIntoView({ block: 'start', behavior: 'smooth' });
  },

  stepDone(ch, k) {
    const sd = Store.get('sd', {}); const arr = sd[ch.id] || [];
    if (!arr.includes(k)) { arr.push(k); sd[ch.id] = arr; Store.set('sd', sd); }
    if (arr.length >= ch.steps.length) this.markDone(ch.id);
    this.renderStepper(ch);
  },

  markDone(id) {
    const d = Store.get('done', {}); if (d[id]) return;
    d[id] = true; Store.set('done', d); this.renderNav();
    const j = document.getElementById('journey'); if (j) this.renderJourney(j);
  },

  renderStepper(ch) {
    const done = this.stepsDone(ch), k = ch.at;
    ch.sec.querySelector('.stepper').innerHTML = ch.steps.map((s, j) =>
      `<button class="sp ${j === k ? 'cur' : ''} ${done.includes(j) ? 'done' : ''}" data-j="${j}" title="${esc(s.t)}"><span class="spn">${done.includes(j) ? '✓' : j + 1}</span><span class="spt">${s.t}</span></button>`).join('');
    ch.sec.querySelectorAll('.sp').forEach(b => b.onclick = () => this.step(ch, +b.dataset.j));
    const n = this.chapters.indexOf(ch), nx = this.chapters[n + 1], last = k === ch.steps.length - 1;
    ch.sec.querySelector('.prev').disabled = k === 0;
    ch.sec.querySelector('.where').textContent = `תחנה ${k + 1} מתוך ${ch.steps.length}`;
    const next = ch.sec.querySelector('.next');
    next.textContent = last ? (nx ? `לפרק הבא: ${nx.short} ←` : 'סוף הדרך 🎉') : 'לתחנה הבאה ←';
    next.disabled = last && !nx;
    next.classList.toggle('pulse', done.includes(k));
  },

  renderNav() {
    const nav = document.getElementById('nav');
    nav.innerHTML = this.chapters.map((c, i) =>
      `<button data-go="${c.id}" class="${c.id === this.cur ? 'active' : ''} ${this.isDone(c.id) ? 'done' : ''}">
        <span class="ni">${this.isDone(c.id) && i > 0 ? '✓' : c.icon}</span><span class="nt">${c.short}</span></button>`).join('');
  },
  renderJourney(j) {
    j.innerHTML = this.chapters.slice(1).map((c, i) =>
      `<a href="#${c.id}" data-go="${c.id}" class="stop ${this.isDone(c.id) ? 'done' : ''}">
        <span class="si">${c.icon}</span><span class="sn">פרק ${i + 1}${this.isDone(c.id) ? ' ✓' : ''}</span>
        <b>${c.short}</b><small>${c.desc || ''}</small></a>`).join('');
  },
};

window.addEventListener('DOMContentLoaded', () => App.start());
