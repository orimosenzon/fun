// DOM overlay: stats, throttle/lift bars, ring score, messages, help, start
// screen. All static text goes through i18n and is re-rendered on language
// toggle (L key / lang buttons).
import { t, getLang, toggleLang } from './i18n.js';

export class HUD {
  constructor() {
    const el = (cls, parent = document.body, tag = 'div') => {
      const d = document.createElement(tag);
      d.className = cls;
      parent.appendChild(d);
      return d;
    };

    this.stats = el('hud panel stats');
    this.bars = el('hud panel bars');
    this.score = el('hud panel score');
    this.msgEl = el('hud msg');
    this._msgTimer = null;
    this.help = el('hud panel help');
    this.start = el('hud start');
    this.touchMode = false;
    this.onLangChange = null;
    this._lastStats = null;

    this.renderStatic();
    this.$ = id => document.getElementById(id);
  }

  // (re)build every static text — called on construction and language toggle
  renderStatic() {
    this.stats.innerHTML = `
      <div class="row"><span>${t('speed')}</span><b id="hud-speed">0</b><span class="u">${t('kmh')}</span></div>
      <div class="row"><span>${t('agl')}</span><b id="hud-agl">0</b><span class="u">${t('m')}</span></div>
      <div class="row"><span>${t('alt')}</span><b id="hud-alt">0</b><span class="u">${t('m')}</span></div>
      <div class="row"><span>${t('vspd')}</span><b id="hud-vspd">0</b><span class="u">${t('ms')}</span></div>
      <div class="row small"><span id="hud-wind"></span></div>
      <div class="row small"><span id="hud-assist"></span></div>`;

    // fill = actual turbine spool; the thin mark = the commanded lever position
    this.bars.innerHTML = `
      <div class="bar-row"><span>${t('throttle')}</span><div class="bar"><div class="fill orange" id="hud-thr"></div><div class="mark" id="hud-thr-mark"></div></div></div>
      <div class="bar-row"><span>${t('lift')}</span><div class="bar"><div class="fill blue" id="hud-col"></div><div class="mark" id="hud-col-mark"></div></div></div>`;

    this.score.innerHTML = `
      <div class="row"><span>${t('rings')}</span><b id="hud-rings">0/0</b></div>
      <div class="row"><span>${t('time')}</span><b id="hud-time">—</b></div>
      <div class="row small"><span id="hud-best"></span></div>`;

    const lines = this.touchMode ? t('touchHelpLines') : t('helpLines');
    const helpWasHidden = this.help.style.display === 'none';
    this.help.innerHTML = `<b>${t('helpTitle')}</b>` + lines.map(l => `<div>${l}</div>`).join('');
    if (helpWasHidden) this.help.style.display = 'none';

    const keys1 = this.touchMode ? t('touchKeys1') : t('startKeys1');
    const keys2 = this.touchMode ? t('touchKeys2') : t('startKeys2');
    const press = this.touchMode ? t('touchPress') : t('press');
    const note = this.touchMode && innerWidth < innerHeight
      ? `<p style="font-size:13px;color:#ffc36b;margin-top:10px;">${t('portraitNote')}</p>` : '';
    this.start.innerHTML = `
      <div class="card">
        <h1>FABLE</h1>
        <h2>${t('subtitle')}</h2>
        <p>${t('intro')}</p>
        <div class="keys">
          <div>${keys1}</div>
          <div>${keys2}</div>
        </div>
        <p class="press">${press}</p>${note}
        <button type="button" class="lang-btn">${t('langBtn')}</button>
      </div>`;
    this.start.querySelector('.lang-btn').addEventListener('click', e => {
      e.stopPropagation();
      this.toggleLang();
    });
    if (this._lastStats) this.setStats(this._lastStats);
  }

  toggleLang() {
    toggleLang();
    this.renderStatic();
    if (this.onLangChange) this.onLangChange(getLang());
  }

  setTouchMode(isTouch) {
    if (!isTouch) return;
    this.touchMode = true;
    // hidden until the '?' button toggles it — little screen room on a phone
    this.help.style.display = 'none';
    this.renderStatic();
  }

  hideStart() { this.start.style.display = 'none'; }
  toggleHelp() { this.help.style.display = this.help.style.display === 'none' ? '' : 'none'; }

  setStats(s) {
    this._lastStats = s;
    this.$('hud-speed').textContent = Math.round(s.speedKmh);
    this.$('hud-agl').textContent = Math.max(0, s.agl).toFixed(0);
    this.$('hud-alt').textContent = s.alt.toFixed(0);
    this.$('hud-vspd').textContent = (s.vspd > 0 ? '+' : '') + (s.vspd ?? 0).toFixed(1);
    this.$('hud-assist').textContent = s.assist ? t('assistOn') : t('assistOff');
    this.$('hud-wind').textContent = s.wind || '';
    this.$('hud-thr').style.width = (Math.min(1, s.spoolRear ?? s.throttle) * 100).toFixed(0) + '%';
    this.$('hud-col').style.width = ((s.spoolLift ?? s.collective) / 1.25 * 100).toFixed(0) + '%';
    this.$('hud-thr-mark').style.insetInlineStart = (s.throttle * 100).toFixed(0) + '%';
    this.$('hud-col-mark').style.insetInlineStart = (s.collective / 1.25 * 100).toFixed(0) + '%';
    this.$('hud-rings').textContent = s.rings;
    this.$('hud-time').textContent = s.time;
    this.$('hud-best').textContent = s.best !== null ? t('best') + s.best.toFixed(1) + ' ' + t('secondsShort') : '';
  }

  message(text, ms = 2000, cls = '') {
    this.msgEl.textContent = text;
    this.msgEl.className = 'hud msg show ' + cls;
    clearTimeout(this._msgTimer);
    this._msgTimer = setTimeout(() => { this.msgEl.className = 'hud msg'; }, ms);
  }

  showCrash(code) {
    this.message(t('crash_' + code) + t('backToPad'), 2400, 'crash');
  }
}
