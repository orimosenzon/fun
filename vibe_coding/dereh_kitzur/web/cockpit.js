/* The F-16's office: the head-up display and the top of the instrument
 * panel, drawn over the flight (18/9/2026).
 *
 * Ori asked for the jet to be seen from inside a real cockpit. What a Viper
 * pilot actually sees is mostly the world: the canopy is one piece of
 * polycarbonate with no frame in front, the seat leans back thirty degrees,
 * the stick is on the right console and the throttle on the left, out of the
 * picture. What is in the picture is the HUD, a glass in front of the eyes
 * with green symbols projected at infinity, and under it the glare shield,
 * the integrated control panel (ICP) with its little data entry display
 * (DED), and the two multifunction displays (MFDs) at the knees. So this is
 * two-dimensional on purpose: a 3D cockpit would cost the phone and hide the
 * town, and the real thing hides nothing.
 *
 * The HUD is conformal, which is the whole point of a HUD: its horizon line
 * lies on the horizon in the picture, the pitch ladder is drawn from the same
 * pinhole camera explore.js gives MapLibre (`F` pixels of focal length, the
 * boresight at the middle of the window), and the ladder rolls with the bank
 * because the picture does. The symbology follows the real layout as the
 * public manuals describe it (Falconpedia, the T.O.-derived sim manuals):
 *
 *   - flight path marker: a circle with wings and a fin, where the aircraft
 *     is actually going; dashed when it has been clamped to the glass;
 *   - pitch ladder in 5 degree steps, solid above the horizon and dashed
 *     below, numbers at the ends with tips pointing towards the horizon,
 *     the horizon line longer and unnumbered;
 *   - airspeed scale on the left, in knots, "C" for calibrated;
 *   - altitude scale on the right, in feet, labels in hundreds, "R" when
 *     the radar altimeter is the source, below 1,500 ft;
 *   - heading scale at the bottom, ticks every 5 degrees, two-digit labels
 *     every 10, the heading in a box above the caret;
 *   - g at the upper left; Mach, the peak g of the flight and the master
 *     mode at the lower left; range to the steerpoint at the lower right;
 *   - the roll scale and its pointer at the bottom, and LIMIT in the middle
 *     when the flight control computer is holding the stick at 9 g.
 *
 * Units are the aircraft's. Israelis think in km/h and metres, and the
 * take-off card says so in one line; a HUD in km/h would be a costume.
 *
 * The panel is an SVG rebuilt on resize, with three live parts: the DED
 * shows the steerpoint (the nearest trail), the left MFD is a horizontal
 * situation display with the trails around the aircraft drawn heading-up on
 * a canvas, and two engine gauges follow the throttle and the burner.
 * explore.js owns the flight and calls paint() every frame with the state;
 * this file only draws. */
'use strict';

const Cockpit = (() => {

  const RAD = Math.PI / 180;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const KT = 1.943844;        // m/s to knots
  const FT = 3.280840;        // metres to feet
  const NM = 1 / 1852;        // metres to nautical miles

  /* ---------- the glass ----------
   *
   * The real HUD covers about twenty degrees of the view, a little below the
   * boresight. Here the boresight is the middle of the window and the camera
   * is pitched ten to thirty degrees below the horizon to look at the town,
   * so the glass is set UP degrees above the middle: skimming low the horizon
   * is inside it, and high up, looking down, the horizon sits at its top edge
   * and the flight path marker is clamped there, dashed, which is what a HUD
   * does when the velocity vector leaves the field of view. */
  const UP = 8;               // degrees the glass centre sits above the boresight
  const HALF_W = 8;           // half the glass, degrees. The real glass spans more, but
  const HALF_H = 6.5;         // so does a pilot's eye: this camera is a 32-degree telephoto
  const SPAN_KT = 90;         // airspeed scale: knots visible above and below the box
  const SPAN_FT = 1000;       // altitude scale: feet
  const SPAN_HDG = 15;        // heading scale: degrees either side
  const RADAR_FT = 1500;      // below this the altitude is the radar altimeter's
  const HSD_RANGE = 6000;     // metres to the outer ring of the situation display
  const LIMIT_G = 8.6;        // where LIMIT comes on, just under the 9 g limiter

  let host = null;            // the .fly-hud it lives in
  let root = null;            // .ckpt
  let svg = null;             // the HUD
  let panel = null;           // the instrument panel
  let hsd = null, hctx = null;// the left MFD's canvas and context
  let ded = null;             // the DED's lines
  let geo = null;             // the layout, redone when the window or the camera changes
  let el = {};                // the live nodes of the HUD, by name
  let lad = [];               // the pitch ladder's rungs
  let maxG = 1;
  let prevAlt = null, vz = 0, prevT = 0;
  let frame = 0;
  let lastHdg = -1, lastKt = -1, lastFt = -1, lastM = '', lastG = '', lastMax = '', lastStpt = '';

  const S = (n) => `${n.toFixed(1)}`;

  /* ---------- building ---------- */

  function mount(hudEl) {
    if (root) return;
    host = hudEl;
    root = document.createElement('div');
    root.className = 'ckpt';
    root.hidden = true;
    root.innerHTML = `
      <svg class="ckpt-hud" aria-hidden="true"></svg>
      <div class="ckpt-panel" aria-hidden="true"></div>`;
    svg = root.querySelector('.ckpt-hud');
    panel = root.querySelector('.ckpt-panel');
    host.appendChild(root);
  }

  /** On, for the jet; off for the others. Resets the flight's peak g. */
  function show(on) {
    if (!root) return;
    root.hidden = !on;
    document.body.classList.toggle('fly-jet', !!on);
    if (on) {
      maxG = 1; prevAlt = null; vz = 0; frame = 0;
      lastHdg = lastKt = lastFt = -1; lastM = lastG = lastMax = lastStpt = '';
      geo = null;   // lay out afresh: the window may have changed since the last flight
    }
  }

  /** The geometry of the glass from the window and the camera's focal length
   *  in pixels; the boresight is the middle of the window. */
  function layout(w, h, F) {
    const ppd = F * RAD;                            // pixels per degree at the boresight
    const fs = clamp(ppd / 27, 0.72, 1.25);         // type scale: 1 on a 900px-high desktop
    const gw = Math.min(2 * HALF_W * ppd, w * 0.92);
    const gh = Math.min(2 * HALF_H * ppd, h * 0.86);
    const cx = w / 2, cy = h / 2;
    const narrow = w < 560;
    let gy = cy - UP * ppd;
    // Never off the top of a short window; on a phone, under the buttons too.
    const top = narrow ? 112 : 8;
    if (gy - gh / 2 < top) gy = top + gh / 2;
    const ph = narrow ? clamp(h * 0.11, 64, 96) : clamp(h * 0.19, 120, 220);
    geo = {
      w, h, F, ppd, fs, cx, cy, gx: cx, gy, gw, gh,
      x0: cx - gw / 2, x1: cx + gw / 2, y0: gy - gh / 2, y1: gy + gh / 2,
      narrow, ph
    };
    document.documentElement.style.setProperty('--ckpt-h', `${ph}px`);
    buildHud();
    buildPanel();
  }

  const svgEl = (tag, attrs, parent) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const txt = (x, y, s, cls, parent, anchor = 'middle') =>
    svgEl('text', { x, y, class: cls, 'text-anchor': anchor }, parent).appendChild(document.createTextNode(s)).parentNode;

  function buildHud() {
    const g = geo;
    svg.setAttribute('viewBox', `0 0 ${g.w} ${g.h}`);
    svg.setAttribute('width', g.w);
    svg.setAttribute('height', g.h);
    svg.style.fontSize = `${13 * g.fs}px`;
    svg.innerHTML = '';
    el = {};

    const defs = svgEl('defs', {}, svg);
    // The ladder stops short of the bottom band, which belongs to the roll
    // scale and the heading tape; a rung through the heading box reads as
    // neither.
    const ladBottom = g.y1 - 0.3 * g.gh;
    g.ladBottom = ladBottom;
    const clip = svgEl('clipPath', { id: 'ckpt-glass' }, defs);
    svgEl('rect', { x: g.x0, y: g.y0, width: g.gw, height: ladBottom - g.y0 }, clip);
    // The scales get their own windows, in two parts each: the band behind
    // the value box is left out, as the real display blanks the scale there.
    const tapeH = g.gh * 0.5;
    const tapeW = 64 * g.fs;
    const band = 11 * g.fs;
    const clipL = svgEl('clipPath', { id: 'ckpt-tape-l' }, defs);
    svgEl('rect', { x: g.x0, y: g.gy - tapeH / 2, width: tapeW, height: tapeH / 2 - band }, clipL);
    svgEl('rect', { x: g.x0, y: g.gy + band, width: tapeW, height: tapeH / 2 - band }, clipL);
    const clipR = svgEl('clipPath', { id: 'ckpt-tape-r' }, defs);
    svgEl('rect', { x: g.x1 - tapeW, y: g.gy - tapeH / 2, width: tapeW, height: tapeH / 2 - band }, clipR);
    svgEl('rect', { x: g.x1 - tapeW, y: g.gy + band, width: tapeW, height: tapeH / 2 - band }, clipR);
    const hdgW = 2 * SPAN_HDG * hdgK();
    const clipH = svgEl('clipPath', { id: 'ckpt-tape-h' }, defs);
    svgEl('rect', { x: g.gx - hdgW / 2, y: g.y1 - 34 * g.fs, width: hdgW, height: 30 * g.fs }, clipH);

    // The glass itself: a whisper of green and an edge, so the eye knows
    // where the symbols will and will not go.
    svgEl('rect', { x: g.x0, y: g.y0, width: g.gw, height: g.gh, rx: 6 * g.fs, class: 'ckpt-pane' }, svg);

    // ---- the pitch ladder, rolled with the picture, clipped to the glass ----
    const ladClip = svgEl('g', { 'clip-path': 'url(#ckpt-glass)' }, svg);
    el.ladder = svgEl('g', { class: 'ckpt-ladder', id: 'hud-ladder' }, ladClip);
    lad = [];
    // The rungs reach 4.2 degrees either side, or less on a phone, where the
    // glass is as wide as the screen and the scales would be under the numbers.
    const room = g.gw / 2 - tapeW;
    const half = Math.min(4.2 * g.ppd, room - 26 * g.fs), gap = 1.3 * g.ppd, tip = 0.5 * g.ppd;
    const hh = Math.min(6 * g.ppd, room - 6 * g.fs), hg = 1.6 * g.ppd;   // the horizon: longer
    for (let phi = -85; phi <= 85; phi += 5) {
      const r = svgEl('g', { class: 'rung' + (phi < 0 ? ' below' : phi === 0 ? ' horizon' : '') }, el.ladder);
      const L = phi === 0 ? hh : half, G = phi === 0 ? hg : gap;
      const d = phi === 0 ? '' : phi > 0 ? `M${-L},0 v${tip} M${L},0 v${tip}` : `M${-L},0 v${-tip} M${L},0 v${-tip}`;
      svgEl('path', { d: `M${-L},0 H${-G} M${G},0 H${L} ${d}`, class: 'rung-line' }, r);
      if (phi !== 0) {
        const s = String(Math.abs(phi));
        const ty = phi > 0 ? 5 * g.fs : -3 * g.fs;
        txt(-L - 4 * g.fs, ty, s, 'rung-num', r, 'end');
        txt(L + 4 * g.fs, ty, s, 'rung-num', r, 'start');
      }
      r.style.display = 'none';
      lad.push({ phi, node: r });
    }

    // ---- the flight path marker: not rolled, the wings are the aircraft's ----
    el.fpm = svgEl('g', { class: 'ckpt-fpm', id: 'hud-fpm' }, svg);
    const fr = 0.36 * g.ppd;
    svgEl('circle', { r: fr, class: 'fpm-ring' }, el.fpm);
    svgEl('path', { d: `M${-fr},0 h${-fr * 1.9} M${fr},0 h${fr * 1.9} M0,${-fr} v${-fr * 1.1}`, class: 'fpm-wing' }, el.fpm);

    // ---- airspeed, left ----
    const spdX = g.x0 + 34 * g.fs;
    const kS = tapeH / (2 * SPAN_KT);
    const tapeL = svgEl('g', { 'clip-path': 'url(#ckpt-tape-l)' }, svg);
    el.spd = svgEl('g', {}, tapeL);
    for (let v = 0; v <= 1400; v += 10) {
      const long = v % 50 === 0;
      svgEl('line', { x1: spdX + 24 * g.fs, y1: -v * kS, x2: spdX + (long ? 12 : 18) * g.fs, y2: -v * kS, class: 'tick' }, el.spd);
      if (long) txt(spdX + 9 * g.fs, -v * kS + 4 * g.fs, String(v), 'tape-num', el.spd, 'end');
    }
    el.spd.dataset.k = kS;
    el.spd.dataset.x = spdX;
    boxL(spdX + 24 * g.fs, g.gy, 'spdBox', 'C');

    // ---- altitude, right ----
    const altX = g.x1 - 34 * g.fs;
    const kA = tapeH / (2 * SPAN_FT);
    const tapeR = svgEl('g', { 'clip-path': 'url(#ckpt-tape-r)' }, svg);
    el.alt = svgEl('g', {}, tapeR);
    for (let f = 0; f <= 52000; f += 100) {
      const long = f % 500 === 0;
      svgEl('line', { x1: altX - 24 * g.fs, y1: -f * kA, x2: altX - (long ? 12 : 18) * g.fs, y2: -f * kA, class: 'tick' }, el.alt);
      if (long) txt(altX - 9 * g.fs, -f * kA + 4 * g.fs, String(f / 100), 'tape-num', el.alt, 'start');
    }
    el.alt.dataset.k = kA;
    boxR(altX - 24 * g.fs, g.gy, 'altBox', 'R');

    // ---- heading, bottom ----
    const hy = g.y1 - 14 * g.fs;
    const kH = hdgK();
    const tapeHd = svgEl('g', { 'clip-path': 'url(#ckpt-tape-h)' }, svg);
    el.hdg = svgEl('g', {}, tapeHd);
    // Three copies, so the window over 355..005 shows both ends.
    for (let rep = -1; rep <= 1; rep++) {
      for (let a = 0; a < 360; a += 5) {
        const x = (a + rep * 360) * kH;
        const long = a % 10 === 0;
        svgEl('line', { x1: x, y1: hy, x2: x, y2: hy - (long ? 9 : 5) * g.fs, class: 'tick' }, el.hdg);
        if (long) txt(x, hy - 12 * g.fs, String(a === 0 ? 36 : a / 10).padStart(2, '0'), 'tape-num', el.hdg);
      }
    }
    el.hdg.dataset.k = kH;
    // The caret and the boxed heading above it.
    svgEl('path', { d: `M${g.gx},${hy - 1} l${-5 * g.fs},${7 * g.fs} h${10 * g.fs} z`, class: 'caret' }, svg);
    const bw = 40 * g.fs, bh = 17 * g.fs;
    svgEl('rect', { x: g.gx - bw / 2, y: hy - 29 * g.fs - bh, width: bw, height: bh, class: 'box' }, svg);
    el.hdgBox = txt(g.gx, hy - 29 * g.fs - 4 * g.fs, '000', 'box-num', svg);
    el.hdgBox.id = 'hud-hdg';
    el.spdBox.id = 'hud-spd';
    el.altBox.id = 'hud-alt';

    // ---- the roll scale: aircraft-fixed ticks, a pointer that shows where down is ----
    const R = g.gh * 0.3;
    const rcy = hy - 48 * g.fs - R;
    el.roll = svgEl('g', {}, svg);
    for (const a of [-45, -30, -20, -10, 0, 10, 20, 30, 45]) {
      const long = a === 0 || Math.abs(a) === 45;
      const len = (long ? 9 : 5) * g.fs;
      const s = Math.sin(a * RAD), c = Math.cos(a * RAD);
      svgEl('line', {
        x1: g.gx + R * s, y1: rcy + R * c, x2: g.gx + (R + len) * s, y2: rcy + (R + len) * c, class: 'tick'
      }, el.roll);
    }
    el.rollPtr = svgEl('path', { d: `M${g.gx},${rcy + R - 2} l${-4.5 * g.fs},${-8 * g.fs} h${9 * g.fs} z`, class: 'caret' }, svg);
    el.rollPtr.dataset.cx = g.gx;
    el.rollPtr.dataset.cy = rcy;

    // ---- the text blocks ----
    el.g = txt(g.x0 + 10 * g.fs, g.y0 + 18 * g.fs, '1.0', 'data', svg, 'start');
    el.g.id = 'hud-g';
    el.mach = txt(g.x0 + 10 * g.fs, g.y1 - 60 * g.fs, '0.00', 'data', svg, 'start');
    el.mach.id = 'hud-mach';
    el.maxG = txt(g.x0 + 10 * g.fs, g.y1 - 44 * g.fs, '1.0', 'data', svg, 'start');
    txt(g.x0 + 10 * g.fs, g.y1 - 28 * g.fs, 'NAV', 'data', svg, 'start');
    el.stpt = txt(g.x1 - 10 * g.fs, g.y1 - 44 * g.fs, '', 'data', svg, 'end');
    el.warn = txt(g.gx, g.gy + 2 * g.ppd, 'LIMIT', 'warn', svg);
    el.warn.style.display = 'none';
  }

  /** Pixels per degree of the heading scale: thirty degrees across a good
   *  part of the glass. */
  const hdgK = () => (geo.gw * 0.62) / (2 * SPAN_HDG);

  /** A value box with a pointer, on the left scale (pointing right) ... */
  function boxL(x, y, name, letter) {
    const g = geo, bw = 46 * g.fs, bh = 18 * g.fs, p = 6 * g.fs;
    svgEl('path', { d: `M${x - bw},${y - bh / 2} h${bw} l${p},${bh / 2} l${-p},${bh / 2} h${-bw} z`, class: 'box' }, svg);
    el[name] = txt(x - 4 * g.fs, y + 4.5 * g.fs, '0', 'box-num', svg, 'end');
    el[name + 'L'] = txt(x - bw - 7 * g.fs, y + 4.5 * g.fs, letter, 'data', svg, 'end');
  }
  /** ... and on the right one (pointing left). */
  function boxR(x, y, name, letter) {
    const g = geo, bw = 56 * g.fs, bh = 18 * g.fs, p = 6 * g.fs;
    svgEl('path', { d: `M${x + bw},${y - bh / 2} h${-bw} l${-p},${bh / 2} l${p},${bh / 2} h${bw} z`, class: 'box' }, svg);
    el[name] = txt(x + 4 * g.fs, y + 4.5 * g.fs, '0', 'box-num', svg, 'start');
    el[name + 'L'] = txt(x - p - 4 * g.fs, y + 4.5 * g.fs, letter, 'data', svg, 'end');
    el[name + 'L'].style.display = 'none';
  }

  /* ---------- the panel ---------- */

  function buildPanel() {
    const g = geo, w = g.w, ph = g.ph, n = g.narrow;
    panel.style.height = `${ph}px`;
    const mw = Math.min(w * 0.22, 260), mh = ph * 0.66, my = ph * 0.34;   // the MFDs, cut by the screen's edge
    const iw = n ? w * 0.72 : Math.min(w * 0.34, 380), ix = w / 2 - iw / 2, iy = n ? ph * 0.18 : ph * 0.3;
    const dedH = n ? ph - iy - 6 : ph * 0.36;
    // The glare shield: a hood with a shallow dome over the ICP, a highlight
    // along its lip where the sun would catch it.
    const lip = `M0,${ph * 0.16} C${w * 0.3},${ph * 0.16} ${w * 0.4},${2} ${w / 2},${2} C${w * 0.6},${2} ${w * 0.7},${ph * 0.16} ${w},${ph * 0.16}`;
    let s = `<svg viewBox="0 0 ${w} ${ph}" width="${w}" height="${ph}" preserveAspectRatio="none">
      <path d="${lip} V${ph} H0 Z" class="hood"/>
      <path d="${lip}" class="lip"/>`;
    if (!n) {
      s += mfd(w * 0.05, my, mw, mh, 'L') + mfd(w * 0.95 - mw, my, mw, mh, 'R');
      // Engine gauges between the ICP and the right MFD: RPM and nozzle
      // position, the two a pilot glances at when the burner lights.
      const gr = Math.min(ph * 0.13, 26), gx1 = ix + iw + (w * 0.95 - mw - ix - iw) / 2;
      s += gauge(gx1 - gr * 1.3, iy + gr * 1.4, gr, 'RPM', 'rpm') + gauge(gx1 + gr * 1.3, iy + gr * 1.4, gr, 'NOZ', 'noz');
      // Caution lights on the other side; MASTER CAUTION is the one that lights.
      const lx = ix - (ix - (w * 0.05 + mw)) / 2 - 42, ly = iy + 8;
      const names = ['MASTER CAUTION', 'TF FAIL', 'ENG FIRE', 'HYD/OIL', 'FLCS', 'CANOPY'];
      names.forEach((t, i) => {
        const x = lx + (i % 2) * 44, y = ly + Math.floor(i / 2) * 18;
        s += `<rect x="${x}" y="${y}" width="40" height="14" rx="2" class="lamp${i === 0 ? ' master' : ''}"/>
              <text x="${x + 20}" y="${y + 9.5}" class="lamp-t">${t}</text>`;
      });
    }
    // The ICP: the DED on top, a row of keys under it, a knob at each end.
    s += `<rect x="${ix}" y="${iy}" width="${iw}" height="${ph - iy + 4}" rx="4" class="icp"/>
          <rect x="${ix + 10}" y="${iy + 6}" width="${iw - 20}" height="${dedH}" rx="2" class="ded"/>`;
    if (!n) {
      const keys = ['COM1', 'COM2', 'IFF', 'LIST', 'A-A', 'A-G'];
      const kw = (iw - 20) / keys.length, ky = iy + dedH + 14;
      keys.forEach((k, i) => {
        s += `<rect x="${ix + 10 + i * kw + 3}" y="${ky}" width="${kw - 6}" height="18" rx="3" class="key"/>
              <text x="${ix + 10 + i * kw + kw / 2}" y="${ky + 12.5}" class="key-t">${k}</text>`;
      });
      s += `<circle cx="${ix - 1}" cy="${ky + 9}" r="7" class="knob"/><circle cx="${ix + iw + 1}" cy="${ky + 9}" r="7" class="knob"/>`;
    }
    s += '</svg>';
    panel.innerHTML = s;

    // The DED's text is HTML, not SVG: it carries a Hebrew trail name and an
    // RTL line is a browser's job.
    ded = document.createElement('div');
    ded.className = 'ckpt-ded';
    ded.style.cssText = `left:${ix + 12}px; top:${iy + 7}px; width:${iw - 24}px; height:${dedH - 2}px; font-size:${clamp(dedH / 4.1, 9, 13)}px;`;
    ded.innerHTML = '<span class="l1" id="hud-ded"></span><strong class="l2" id="hud-stpt"></strong><span class="l3" id="hud-hint"></span>';
    panel.appendChild(ded);

    if (!n) {
      // The HSD, on the left MFD's screen.
      const sx = w * 0.05 + 22, sy = my + 20, sw = mw - 44, sh = mh - 20;
      hsd = document.createElement('canvas');
      hsd.className = 'ckpt-hsd';
      const dpr = Math.min(devicePixelRatio || 1, 2);
      hsd.width = Math.round(sw * dpr); hsd.height = Math.round(sh * dpr);
      hsd.style.cssText = `left:${sx}px; top:${sy}px; width:${sw}px; height:${sh}px;`;
      panel.appendChild(hsd);
      hctx = hsd.getContext('2d');
      hctx.scale(dpr, dpr);
      // The right MFD: the radar page, at rest. Text only; it is a picture.
      const rx = w * 0.95 - mw + 22, ry = my + 20;
      const fcr = document.createElement('div');
      fcr.className = 'ckpt-fcr';
      fcr.style.cssText = `left:${rx}px; top:${ry}px; width:${sw}px; height:${sh}px;`;
      fcr.innerHTML = '<span>FCR</span><span>NAV</span><span>RWS</span><span>NORM</span><span>40</span><span>SWAP</span>';
      panel.appendChild(fcr);
    } else {
      hsd = null; hctx = null;
    }
    el.rpm = panel.querySelector('#hud-rpm');
    el.noz = panel.querySelector('#hud-noz');
    el.master = panel.querySelector('.lamp.master');
  }

  function mfd(x, y, w, h, side) {
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h + 10}" rx="6" class="bezel"/>
             <rect x="${x + 22}" y="${y + 20}" width="${w - 44}" height="${h - 20}" class="screen"/>`;
    for (let i = 0; i < 5; i++) {
      s += `<rect x="${x + 30 + i * (w - 60) / 4 - 6}" y="${y + 6}" width="12" height="8" rx="1.5" class="key"/>`;
    }
    for (let i = 0; i < 4; i++) {
      const yy = y + 34 + i * (h - 40) / 3;
      s += `<rect x="${x + 6}" y="${yy}" width="10" height="9" rx="1.5" class="key"/>
            <rect x="${x + w - 16}" y="${yy}" width="10" height="9" rx="1.5" class="key"/>`;
    }
    return s;
  }

  function gauge(cx, cy, r, label, id) {
    let s = `<circle cx="${cx}" cy="${cy}" r="${r}" class="gauge"/>`;
    for (let a = -135; a <= 135; a += 45) {
      const sn = Math.sin(a * RAD), c = -Math.cos(a * RAD);
      s += `<line x1="${cx + sn * r * 0.72}" y1="${cy + c * r * 0.72}" x2="${cx + sn * r * 0.9}" y2="${cy + c * r * 0.9}" class="gauge-tick"/>`;
    }
    s += `<line id="hud-${id}" x1="${cx}" y1="${cy + r * 0.18}" x2="${cx}" y2="${cy - r * 0.8}" data-cx="${cx}" data-cy="${cy}" class="needle" transform="rotate(-135 ${cx} ${cy})"/>
          <circle cx="${cx}" cy="${cy}" r="${r * 0.12}" class="needle-hub"/>
          <text x="${cx}" y="${cy + r * 0.55}" class="gauge-t">${label}</text>`;
    return s;
  }

  /* ---------- painting ---------- */

  /** One frame. `s` is the flight as explore.js has it:
   *  w, h: the window; F: focal length in pixels; pitch: the map's, degrees
   *  from straight down; bank: right wing down positive; bearing; speed (m/s);
   *  alt (m); mach; gee; throttle, burn (0..1); nearest (the steerpoint: name,
   *  d in metres, media: whether it has pictures); own [x, y] and world (the
   *  trails in local metres) for the HSD. */
  function paint(s) {
    if (!root || root.hidden) return;
    if (!geo || geo.w !== s.w || geo.h !== s.h || Math.abs(geo.F - s.F) > 0.5) layout(s.w, s.h, s.F);
    const g = geo;
    const now = performance.now();

    // The vertical speed, from the altitude, smoothed: the flight model has
    // no vz of its own, the arrows move the altitude directly.
    if (prevAlt !== null) {
      const dt = clamp((now - prevT) / 1000, 0.001, 0.1);
      vz += ((s.alt - prevAlt) / dt - vz) * clamp(6 * dt, 0, 1);
    }
    prevAlt = s.alt; prevT = now;

    const v = Math.abs(s.speed);
    const a0 = s.pitch - 90;                 // the nose against the horizon, negative is down
    const gamma = Math.atan2(vz, Math.max(v, 1)) / RAD;

    // ---- ladder ----
    el.ladder.setAttribute('transform', `rotate(${(-s.bank).toFixed(2)} ${g.cx} ${g.cy})`);
    for (const r of lad) {
      const d = r.phi - a0;
      let show = Math.abs(d) < 40;
      let y = 0;
      if (show) {
        y = g.cy - s.F * Math.tan(d * RAD);
        // Rolled, a rung well outside the glass can swing into it; the clip
        // decides, this only spares the far ones.
        show = y > g.y0 - g.gh && y < g.y1 + g.gh;
      }
      r.node.style.display = show ? '' : 'none';
      if (show) r.node.setAttribute('transform', `translate(${g.cx} ${y.toFixed(1)})`);
    }

    // ---- flight path marker ----
    {
      const d = gamma - a0;
      let fy = g.cy - s.F * Math.tan(clamp(d, -80, 80) * RAD);
      // Rolled about the boresight, like the picture.
      const a = -s.bank * RAD, dy = fy - g.cy;
      let x = g.cx - dy * Math.sin(a), y = g.cy + dy * Math.cos(a);
      // Kept to the ladder's part of the glass: below it are the roll scale
      // and the heading tape, and a marker parked on the heading box is two
      // symbols saying nothing.
      const m = 0.6 * g.ppd;
      const lim = x < g.x0 + m || x > g.x1 - m || y < g.y0 + m || y > g.ladBottom - m;
      x = clamp(x, g.x0 + m, g.x1 - m); y = clamp(y, g.y0 + m, g.ladBottom - m);
      el.fpm.setAttribute('transform', `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
      el.fpm.classList.toggle('lim', lim);
    }

    // ---- scales ----
    const kt = v * KT, ft = s.alt * FT, hdg = ((s.bearing % 360) + 360) % 360;
    el.spd.setAttribute('transform', `translate(0 ${(g.gy + kt * el.spd.dataset.k).toFixed(1)})`);
    el.alt.setAttribute('transform', `translate(0 ${(g.gy + ft * el.alt.dataset.k).toFixed(1)})`);
    el.hdg.setAttribute('transform', `translate(${(g.gx - hdg * el.hdg.dataset.k).toFixed(1)} 0)`);
    const ktR = Math.round(kt), ftR = Math.round(ft / 10) * 10, hdgR = Math.round(hdg) % 360;
    if (ktR !== lastKt) { lastKt = ktR; el.spdBox.textContent = String(ktR); }
    if (ftR !== lastFt) {
      lastFt = ftR;
      el.altBox.textContent = ftR.toLocaleString('en-US');
      el.altBoxL.style.display = ft < RADAR_FT ? '' : 'none';
    }
    if (hdgR !== lastHdg) { lastHdg = hdgR; el.hdgBox.textContent = String(hdgR === 0 ? 360 : hdgR).padStart(3, '0'); }

    // ---- roll pointer ----
    el.rollPtr.setAttribute('transform', `rotate(${(-s.bank).toFixed(2)} ${el.rollPtr.dataset.cx} ${el.rollPtr.dataset.cy})`);

    // ---- the numbers ----
    if (s.gee > maxG) maxG = s.gee;
    const gS = S(s.gee), mS = s.mach.toFixed(2), xS = S(maxG);
    if (gS !== lastG) { lastG = gS; el.g.textContent = gS; }
    if (mS !== lastM) { lastM = mS; el.mach.textContent = mS; }
    if (xS !== lastMax) { lastMax = xS; el.maxG.textContent = xS; }
    const st = s.nearest ? `${(s.nearest.d * NM).toFixed(1)}NM` : '';
    if (st !== lastStpt) { lastStpt = st; el.stpt.textContent = st; }
    const limit = s.gee > LIMIT_G;
    el.warn.style.display = limit && (frame & 16) ? '' : 'none';
    if (el.master) el.master.classList.toggle('lit', limit);

    // ---- the panel's live parts ----
    if (el.rpm) {
      // Idle is 70 %, military 100 %; the burner does not turn the fan any
      // faster, it opens the nozzle.
      const rpm = 0.7 + 0.3 * clamp(s.throttle / 0.66, 0, 1);
      el.rpm.setAttribute('transform', `rotate(${(-135 + 270 * rpm).toFixed(1)} ${el.rpm.dataset.cx} ${el.rpm.dataset.cy})`);
      const noz = 0.15 + 0.85 * s.burn;
      el.noz.setAttribute('transform', `rotate(${(-135 + 270 * noz).toFixed(1)} ${el.noz.dataset.cx} ${el.noz.dataset.cy})`);
    }
    if ((frame & 3) === 0) {
      const l1 = ded.querySelector('.l1'), l2 = ded.querySelector('.l2'), l3 = ded.querySelector('.l3');
      l1.textContent = `HDG ${String(hdgR === 0 ? 360 : hdgR).padStart(3, '0')}  ${ktR} KT  ${ftR.toLocaleString('en-US')} FT`;
      if (s.nearest && s.nearest.name) {
        l2.textContent = s.nearest.name;
        l3.textContent = s.nearest.media ? 'Enter לתמונות · ⇧Enter לפתיחה' : '⇧Enter לפתיחה';
      } else {
        l2.textContent = 'STPT  ---';
        l3.textContent = '';
      }
    }
    if (hctx && (frame & 3) === 1) paintHsd(s);
    frame++;
  }

  /** The horizontal situation display: heading up, the aircraft low on the
   *  screen so most of the range is ahead, range rings at half and full, and
   *  the trails around drawn as lines, the revealed ones bright. */
  function paintHsd(s) {
    const c = hctx, W = hsd.clientWidth, H = hsd.clientHeight;
    const ox = W / 2, oy = H * 0.68, ppm = (H * 0.62) / HSD_RANGE;
    c.clearRect(0, 0, W, H);
    c.save();
    c.beginPath(); c.rect(0, 0, W, H); c.clip();
    c.strokeStyle = 'rgba(90,220,120,.35)';
    c.lineWidth = 1;
    for (const r of [0.5, 1]) { c.beginPath(); c.arc(ox, oy, HSD_RANGE * r * ppm, 0, Math.PI * 2); c.stroke(); }
    // The reveal reach, what the flight lights up around it.
    c.strokeStyle = 'rgba(90,220,120,.18)';
    c.setLineDash([3, 3]);
    c.beginPath(); c.arc(ox, oy, clamp(s.alt * 5, 420, 3000) * ppm, 0, Math.PI * 2); c.stroke();
    c.setLineDash([]);
    c.fillStyle = 'rgba(120,255,140,.7)';
    c.font = '9px ui-monospace, Menlo, Consolas, monospace';
    c.fillText(`${(HSD_RANGE * NM).toFixed(0)}NM`, 4, 11);
    c.textAlign = 'right';
    c.fillText(`${Math.round(s.bearing) % 360}°`, W - 4, 11);
    c.textAlign = 'left';

    const b = -s.bearing * RAD, sb = Math.sin(b), cb = Math.cos(b);
    const [px, py] = s.own;
    const P = (x, y) => {                 // world metres (east, north) to screen, heading up
      const dx = x - px, dy = y - py;
      return [ox + (dx * cb - dy * sb) * ppm, oy - (dx * sb + dy * cb) * ppm];
    };
    for (const e of s.world) {
      const g = e.g || 0;
      const first = e.xy[0];
      if (Math.hypot(first[0] - px, first[1] - py) > HSD_RANGE * 1.6) continue;
      c.strokeStyle = c.fillStyle = `rgba(120,255,140,${(0.28 + 0.72 * g).toFixed(2)})`;
      c.lineWidth = g > 0.3 ? 1.6 : 1;
      if (e.line) {
        c.beginPath();
        e.xy.forEach((q, i) => { const [x, y] = P(q[0], q[1]); if (i) c.lineTo(x, y); else c.moveTo(x, y); });
        c.stroke();
      } else {
        const [x, y] = P(first[0], first[1]);
        c.fillRect(x - 1.5, y - 1.5, 3, 3);
      }
    }
    // Own ship: a little aircraft, nose up the screen.
    c.strokeStyle = '#d8ffe0';
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(ox, oy - 7); c.lineTo(ox, oy + 5);
    c.moveTo(ox - 6, oy); c.lineTo(ox + 6, oy);
    c.moveTo(ox - 3, oy + 4); c.lineTo(ox + 3, oy + 4);
    c.stroke();
    c.restore();
  }

  return { mount, show, paint };
})();
