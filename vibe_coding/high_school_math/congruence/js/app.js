'use strict';
/* ניווט בין פרקים ושמירת התקדמות */

const Store = {
  key: 'omer-congruence-v1',
  data: {},
  load() { try { this.data = JSON.parse(localStorage.getItem(this.key)) || {}; } catch (e) { this.data = {}; } },
  save() { try { localStorage.setItem(this.key, JSON.stringify(this.data)); } catch (e) { /* בלי שמירה */ } },
  get(k, d) { return k in this.data ? this.data[k] : d; },
  set(k, v) { this.data[k] = v; this.save(); },
};

const App = {
  chapters: [
    { id: 'ch0', short: 'פתיחה', icon: '👋', desc: '' },
    { id: 'ch1', short: 'מה זה חפיפה', icon: '✋', desc: 'להניח משולש על משולש: הזזה, סיבוב והיפוך' },
    { id: 'ch2', short: 'משחק הרמאי', icon: '😈', desc: 'אילו 3 נתונים מספיקים כדי לקבוע משולש?' },
    { id: 'ch3', short: 'מעבדת בנייה', icon: '🔧', desc: 'בונים משולשים ממקלות וזוויות ומבינים למה' },
    { id: 'ch4', short: 'הוכחות', icon: '🧠', desc: 'משתמשים בחפיפה כדי להוכיח דברים חדשים' },
    { id: 'ch5', short: 'תרגול', icon: '🎯', desc: 'איזה משפט? שאלות בלי סוף' },
    { id: 'ch6', short: 'דף סיכום', icon: '📋', desc: 'הכל בדף אחד' },
  ],
  inits: {},
  inited: {},
  current: null,

  start() {
    Store.load();
    this.renderNav();
    this.renderJourney();
    document.addEventListener('click', e => {
      const b = e.target.closest('[data-go]');
      if (b) { e.preventDefault(); this.go(b.dataset.go); }
    });
    window.addEventListener('hashchange', () => { const id = location.hash.slice(1); if (id !== this.current && this.byId(id)) this.go(id); });
    const h = location.hash.slice(1);
    this.go(this.byId(h) ? h : Store.get('last', 'ch0'));
  },
  byId(id) { return this.chapters.find(c => c.id === id); },

  go(id) {
    this.current = id;
    document.querySelectorAll('.chapter').forEach(s => { s.hidden = s.id !== id; });
    document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b.dataset.go === id));
    if (!this.inited[id] && this.inits[id]) { this.inited[id] = true; this.inits[id](); }
    Store.set('last', id);
    if (location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
    if (id === 'ch0') this.markDone('ch0');
    window.scrollTo({ top: 0 });
  },

  isDone(id) { return !!Store.get('done', {})[id]; },
  markDone(id) {
    const d = Store.get('done', {});
    if (d[id]) return;
    d[id] = true; Store.set('done', d);
    this.renderNav(); this.renderJourney();
  },

  renderNav() {
    const nav = document.getElementById('nav');
    nav.innerHTML = this.chapters.map((c, i) =>
      `<button data-go="${c.id}" class="${c.id === this.current ? 'active' : ''} ${this.isDone(c.id) ? 'done' : ''}">
        <span class="ni">${this.isDone(c.id) && i > 0 ? '✓' : c.icon}</span><span class="nt">${c.short}</span></button>`).join('');
  },
  renderJourney() {
    const j = document.getElementById('journey');
    if (!j) return;
    j.innerHTML = this.chapters.slice(1).map((c, i) =>
      `<a href="#${c.id}" data-go="${c.id}" class="stop ${this.isDone(c.id) ? 'done' : ''}">
        <span class="si">${c.icon}</span><span class="sn">פרק ${i + 1}${this.isDone(c.id) ? ' ✓' : ''}</span>
        <b>${c.short}</b><small>${c.desc}</small></a>`).join('');
  },
  // כפתור "לפרק הבא" בסוף פרק
  nextButton(id) {
    const i = this.chapters.findIndex(c => c.id === id), n = this.chapters[i + 1];
    return n ? `<button data-go="${n.id}" class="btn primary">לפרק הבא: ${n.short} ←</button>` : '';
  },
};

window.addEventListener('DOMContentLoaded', () => App.start());
