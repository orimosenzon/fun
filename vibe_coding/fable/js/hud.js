// DOM overlay: stats, throttle/lift bars, ring score, messages, help, start screen.
export class HUD {
  constructor() {
    const el = (cls, parent = document.body, tag = 'div') => {
      const d = document.createElement(tag);
      d.className = cls;
      parent.appendChild(d);
      return d;
    };

    this.stats = el('hud panel stats');
    this.stats.innerHTML = `
      <div class="row"><span>מהירות</span><b id="hud-speed">0</b><span class="u">קמ"ש</span></div>
      <div class="row"><span>גובה מהקרקע</span><b id="hud-agl">0</b><span class="u">מ'</span></div>
      <div class="row"><span>רום</span><b id="hud-alt">0</b><span class="u">מ'</span></div>
      <div class="row small"><span id="hud-assist"></span></div>`;

    this.bars = el('hud panel bars');
    this.bars.innerHTML = `
      <div class="bar-row"><span>מצערת</span><div class="bar"><div class="fill orange" id="hud-thr"></div></div></div>
      <div class="bar-row"><span>עילוי</span><div class="bar"><div class="fill blue" id="hud-col"></div></div></div>`;

    this.score = el('hud panel score');
    this.score.innerHTML = `
      <div class="row"><span>טבעות</span><b id="hud-rings">0/0</b></div>
      <div class="row"><span>זמן</span><b id="hud-time">—</b></div>
      <div class="row small"><span id="hud-best"></span></div>`;

    this.msgEl = el('hud msg');
    this._msgTimer = null;

    this.help = el('hud panel help');
    this.help.innerHTML = `
      <b>שליטה</b>
      <div>← / → — הטיה ופנייה</div>
      <div>↑ / ↓ — הרמת / הורדת האף (עלרוד)</div>
      <div>W / S — מצערת הסילון האחורי</div>
      <div>E / D — עוצמת צינורות העילוי</div>
      <div>רווח — בוסט לסילון האחורי</div>
      <div>Shift — בוסט לצינור המרכזי (עילוי בכיוון הגוף)</div>
      <div>Z / X — גלגול שמאלה / ימינה (בוסט צד!)</div>
      <div>C — מצלמה · R — איפוס · T — מייצב · M — שקט · H — עזרה</div>`;

    this.start = el('hud start');
    this.start.innerHTML = `
      <div class="card">
        <h1>FABLE</h1>
        <h2>אופנוע סילון מעופף</h2>
        <p>מנוע סילון אחורי דוחף קדימה, שלושה צינורות מופנים מטה מרחפים אותך באוויר.<br>
        טוס דרך כל <span class="orange-t">הטבעות הכתומות</span> מהר ככל האפשר — ואל תתרסק.</p>
        <div class="keys">
          <div>←→ היגוי &nbsp;·&nbsp; ↑↓ עלרוד (אף) &nbsp;·&nbsp; WS מצערת &nbsp;·&nbsp; ED עילוי</div>
          <div>רווח בוסט &nbsp;·&nbsp; Shift עילוי־בוסט &nbsp;·&nbsp; Z/X גלגול</div>
        </div>
        <p class="press">לחץ על מקש כלשהו כדי להמריא</p>
      </div>`;

    this.$ = id => document.getElementById(id);
  }

  hideStart() { this.start.style.display = 'none'; }
  toggleHelp() { this.help.style.display = this.help.style.display === 'none' ? '' : 'none'; }

  setStats(s) {
    this.$('hud-speed').textContent = Math.round(s.speedKmh);
    this.$('hud-agl').textContent = Math.max(0, s.agl).toFixed(0);
    this.$('hud-alt').textContent = s.alt.toFixed(0);
    this.$('hud-assist').textContent = s.assist ? 'מייצב טיסה: פועל' : 'מייצב טיסה: כבוי';
    this.$('hud-thr').style.width = (s.throttle * 100).toFixed(0) + '%';
    this.$('hud-col').style.width = (s.collective / 1.25 * 100).toFixed(0) + '%';
    this.$('hud-rings').textContent = s.rings;
    this.$('hud-time').textContent = s.time;
    this.$('hud-best').textContent = s.best ? 'שיא: ' + s.best : '';
  }

  message(text, ms = 2000, cls = '') {
    this.msgEl.textContent = text;
    this.msgEl.className = 'hud msg show ' + cls;
    clearTimeout(this._msgTimer);
    this._msgTimer = setTimeout(() => { this.msgEl.className = 'hud msg'; }, ms);
  }

  showCrash(reason) {
    this.message(reason + ' — חוזרים למשטח...', 2400, 'crash');
  }
}
