// Boot, the two views (map and flight), and the control panel around them.
import { THREE } from './foehn.js';
import { Field, Tone } from './core/field.js';
import { FieldPass, Quad } from './gpu/fieldpass.js';
import { PaletteTex, PALETTES, paletteCss } from './gpu/palette.js';
import { MapView, HOME, HOME_SCALE } from './map.js';
import { bp, bitsFor, bpToD, bpSubD, fixToDec } from './core/fixed.js';

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const showErr = (e) => { const el = $('err'); el.style.display = 'block'; el.textContent += (e && e.stack || e) + '\n'; console.error(e); };
addEventListener('error', (e) => showErr(e.error || e.message)); addEventListener('unhandledrejection', (e) => showErr(e.reason));

const MOBILE = matchMedia('(pointer: coarse)').matches || Math.min(screen.width, screen.height) < 600;
const KEY = 'mandelscape.v1';
const saved = (() => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } })();
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch { /* private mode */ } };

const PRESETS = [
  { name: 'עמק סוסוני הים', x: '-0.74453986046', y: '0.12172377365', scale: 3e-4 },
  { name: 'עמק הפילים', x: '0.28693186889', y: '0.01428669390', scale: 2e-6 },
  { name: 'ספירלה משולשת', x: '-0.08868132570', y: '0.65450583520', scale: 4e-6 },
  { name: 'כוכב ים', x: '-0.37400413900', y: '0.65979217500', scale: 6e-6 },
  { name: 'מנדלברוט זעיר על המחט', x: '-1.78644025550', y: '0', scale: 4e-9 },
  { name: 'ספירלת סוסון עמוקה', x: '-0.743643887037158704752191506114774', y: '0.131825904205311970493132056385139', scale: 1.2e-13 },
  { name: 'לב הספירלה: צלילת ענק', x: '-0.743643887037158704752191506114774', y: '0.131825904205311970493132056385139', scale: 1e-26 },
];

function boot() {
  const canvas = $('c');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false, alpha: false });
  renderer.autoClear = false; renderer.setPixelRatio(1);
  if (!renderer.capabilities.isWebGL2 || !renderer.extensions.get('EXT_color_buffer_float')) {
    document.body.innerHTML = '<p style="padding:40px;font-family:sans-serif;color:#fff">הדפדפן או כרטיס המסך אינם תומכים ברינדור הדרוש (WebGL2 עם מאגרי נקודה צפה). נסו דפדפן עדכני.</p>';
    return;
  }
  const field = new Field(), tone = new Tone();
  const app = {
    renderer, canvas, field, tone, fp: new FieldPass(renderer, field), quad: new Quad(renderer), pal: new PaletteTex(),
    toneU: { value: new THREE.Vector4(2, 6, 0.55, 0) },
    opt: { quality: saved.quality || (MOBILE ? 0.8 : 1), cap: MOBILE ? 20000 : 100000, mobile: MOBILE, floatLinear: !!renderer.extensions.get('OES_texture_float_linear') && !Q.has('nolinear') },
    cycle: false, colOffset: 0, density: saved.density || 0.55, relief: saved.relief ?? 0.35, edge: 0.9, slope: saved.slope || 0.5,
    syncTone() { this.toneU.value.set(tone.lo, tone.w, this.density * (this.flying ? 1.6 : 1), this.colOffset); },     // (on the land the bands are also steps: a few more of them)
    $, saved, persist, Q,
  };
  const map = new MapView(app);
  let fly = null, mode = 'map', viewW = 1, viewH = 1;
  app.pal.set(Math.min(PALETTES.length - 1, saved.palette | 0));

  // ---------- size ----------
  function resize() {
    viewW = canvas.clientWidth; viewH = canvas.clientHeight;
    if (mode === 'map') {
      const dpr = Math.min(devicePixelRatio || 1, MOBILE ? 1.5 : 2) * (+Q.get('scale') || 1);
      const w = Math.max(2, Math.round(viewW * dpr)), h = Math.max(2, Math.round(viewH * dpr));
      renderer.setSize(w, h, false); map.resize(w, h);
    } else if (fly) fly.resize(viewW, viewH);
    drawOrbit();
  }
  addEventListener('resize', resize);
  if (window.visualViewport) visualViewport.addEventListener('resize', resize);

  // ---------- status line, link ----------
  const coordsEl = $('coords');
  const fmtZoom = (z) => (z < 1000 ? '×' + z.toFixed(1) : '×10^' + Math.log10(z).toFixed(1));
  let hashT = 0;
  const place = () => (mode === 'fly' && fly ? fly.place() : { center: map.state.center, scale: map.state.scale });
  const linkOf = () => {
    const p = place(), dg = Math.min(40, Math.max(12, Math.ceil(-Math.log10(p.scale)) + 5));
    return `x=${fixToDec(p.center.x, p.center.bits, dg)}&y=${fixToDec(p.center.y, p.center.bits, dg)}&s=${p.scale.toExponential(3)}&p=${app.pal.index}${mode === 'fly' ? '&m=fly' : ''}`;
  };
  const touchHash = () => { clearTimeout(hashT); hashT = setTimeout(() => { try { history.replaceState(null, '', '#' + linkOf()); } catch { /* file:// */ } }, 700); };
  map.onStatus = () => {
    const c = map.coords();
    coordsEl.textContent = `x = ${c.x}   y = ${c.y}   ${fmtZoom(c.zoom)}   ${map.iter} איטרציות`;
    touchHash(); drawOrbit();
  };
  app.setCoords = (txt) => { coordsEl.textContent = txt; touchHash(); };
  app.fmtZoom = fmtZoom;

  // ---------- panel ----------
  const panel = $('panel');
  $('togglePanel').onclick = () => panel.classList.toggle('hidden');
  if (MOBILE) panel.classList.add('hidden');
  const blurAfter = (el) => el.addEventListener('pointerup', () => el.blur());      // or the arrow keys would move a slider instead of the view
  document.querySelectorAll('#panel input, #panel button').forEach(blurAfter);

  for (const p of PRESETS) {
    const b = document.createElement('button'); b.className = 'preset';
    b.innerHTML = `<span>${p.name}</span><span class="z num">${(HOME_SCALE / p.scale).toExponential(0).replace('e+', '×10^')}</span>`;
    b.onclick = () => {
      if (map.recording) return;
      if (MOBILE) panel.classList.add('hidden');
      const pt = bp(p.x, p.y, bitsFor(p.scale));
      if (mode === 'fly') { fly.jump(pt, p.scale); return; }
      map.flyTo(pt, p.scale, Math.max(2500, Math.abs(Math.log10(p.scale / map.state.scale)) * 900));
    };
    blurAfter(b); $('presets').appendChild(b);
  }
  PALETTES.forEach((p, i) => {
    const el = document.createElement('div'); el.className = 'pal' + (i === app.pal.index ? ' active' : ''); el.title = p.name; el.style.background = paletteCss(p);
    el.onclick = () => { document.querySelectorAll('.pal').forEach((x) => x.classList.remove('active')); el.classList.add('active'); app.pal.set(i); saved.palette = i; persist(); map.recolor = true; touchHash(); };
    $('palettes').appendChild(el);
  });
  const slider = (id, key, fmt, on) => {
    const el = $(id), out = $(id + 'Out');
    el.value = app[key] ?? app.opt[key];
    const show = () => { if (out) out.textContent = fmt(+el.value); };
    el.oninput = () => { on(+el.value); saved[key] = +el.value; persist(); show(); };
    show();
  };
  slider('density', 'density', (v) => v.toFixed(2), (v) => { app.density = v; map.recolor = true; });
  slider('relief', 'relief', (v) => Math.round(v * 100) + '%', (v) => { app.relief = v; map.recolor = true; });
  slider('quality', 'quality', (v) => '×' + v.toFixed(1), (v) => { app.opt.quality = v; map.dirty = true; });
  slider('slope', 'slope', (v) => Math.round(Math.atan(v) * 180 / Math.PI) + '°', (v) => { app.slope = v; if (fly) fly.setSlope(v); });
  $('dur').oninput = () => { $('durOut').textContent = $('dur').value; };
  $('cycle').onchange = (e) => { app.cycle = e.target.checked; };
  $('inv').checked = !!saved.invert; $('inv').onchange = (e) => { saved.invert = e.target.checked; persist(); if (fly) fly.input.invert = saved.invert; };
  $('snd').checked = saved.sound !== false; $('snd').onchange = (e) => { saved.sound = e.target.checked; persist(); if (fly) fly.audio.setMuted(!saved.sound); };
  $('fineChk').checked = !!saved.fine; $('fineChk').onchange = (e) => { saved.fine = e.target.checked; persist(); if (fly) fly.setFine(saved.fine); };

  $('resetBtn').onclick = () => { if (map.recording) return; if (mode === 'fly') fly.reset(); else map.flyTo(HOME.center, HOME.scale, 2200); };
  $('fsBtn').onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); };
  if (!document.documentElement.requestFullscreen) $('fsBtn').style.display = 'none';
  $('snapBtn').onclick = () => {
    if (mode === 'fly') fly.draw(); else map.paint();
    canvas.toBlob((blob) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = mode === 'fly' ? 'mandelbrot-flight.png' : 'mandelbrot.png'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }, 'image/png');
  };
  $('linkBtn').onclick = async () => {
    const url = location.origin + location.pathname + '#' + linkOf(), b = $('linkBtn'), t0 = b.textContent;
    try { await navigator.clipboard.writeText(url); b.textContent = 'הקישור הועתק'; } catch { prompt('', url); }
    setTimeout(() => (b.textContent = t0), 1800);
  };
  $('helpBtn').onclick = () => $('helpOverlay').classList.add('show');
  $('helpClose').onclick = () => { $('helpOverlay').classList.remove('show'); saved.helped = true; persist(); };

  // ---------- the two views ----------
  let switching = false;
  async function setMode(m) {
    if (m === mode || switching || map.recording) return;
    switching = true;
    const from = place();
    $('fade').classList.add('on');
    await new Promise((r) => setTimeout(r, 280));
    try {
      if (m === 'fly') {
        if (!fly) { const mod = await import('./fly.js'); fly = new mod.FlyView(app); app.fly = fly; fly.input.invert = !!saved.invert; fly.audio.setMuted(saved.sound === false); if (saved.fine) fly.setFine(true); }
        map.active = false; mode = 'fly'; app.flying = true;
        document.body.className = 'fly'; $('mFly').classList.add('on'); $('mMap').classList.remove('on');
        resize(); fly.enter(from.center, from.scale);
        if (!saved.helped && !Q.has('nohelp')) $('helpOverlay').classList.add('show');
      } else {
        fly.leave(); mode = 'map'; map.active = true; app.flying = false;
        document.body.className = 'map'; $('mMap').classList.add('on'); $('mFly').classList.remove('on');
        resize(); map.jump(from.center, Math.min(from.scale, HOME_SCALE));
      }
    } finally { switching = false; setTimeout(() => $('fade').classList.remove('on'), 60); }
  }
  app.setMode = setMode;
  $('mMap').onclick = $('toMap').onclick = () => setMode('map');
  $('mFly').onclick = $('flyHere').onclick = () => setMode('fly');
  addEventListener('keydown', (e) => {
    if (e.repeat || e.target.tagName === 'INPUT') return;
    if (e.code === 'KeyM') setMode(mode === 'map' ? 'fly' : 'map');
    else if (e.code === 'Escape') { $('helpOverlay').classList.remove('show'); }
  });

  // ---------- the orbit of the point under the pointer (map) ----------
  const oc = $('orbit'), octx = oc.getContext('2d');
  let orbitOn = false, mouse = null;
  $('orbitChk').onchange = (e) => { orbitOn = e.target.checked; oc.style.display = orbitOn ? 'block' : 'none'; drawOrbit(); };
  canvas.addEventListener('pointermove', (e) => { mouse = [e.clientX, e.clientY]; if (orbitOn) drawOrbit(); });
  function drawOrbit() {
    if (!orbitOn || mode !== 'map') { if (oc.width) octx.clearRect(0, 0, oc.width, oc.height); return; }
    const d = devicePixelRatio || 1, W = Math.round(viewW * d), H = Math.round(viewH * d);
    if (oc.width !== W || oc.height !== H) { oc.width = W; oc.height = H; }
    octx.clearRect(0, 0, W, H);
    if (!mouse) return;
    const s = map.state.scale, [cx, cy] = bpToD(map.state.center), [ox, oy] = map.offsetAt(mouse[0], mouse[1]);
    octx.font = `${13 * d}px Heebo, sans-serif`; octx.fillStyle = '#fff'; octx.textAlign = 'left'; octx.direction = 'rtl';
    if (s < 1e-11) { octx.fillText('המסלול מוצג רק עד הגדלה של 10 בחזקת 11', 16 * d, H - 52 * d); return; }
    const x0 = cx + ox, y0 = cy + oy, X = (x) => W / 2 + (x - cx) / s * H / 2, Y = (y) => H / 2 - (y - cy) / s * H / 2;
    let zx = 0, zy = 0, n = 0; const pts = [[X(0), Y(0)]];
    for (; n < 600; n++) { const t = zx * zx - zy * zy + x0; zy = 2 * zx * zy + y0; zx = t; pts.push([X(zx), Y(zy)]); if (zx * zx + zy * zy > 16) break; }
    octx.lineWidth = 1.2 * d; octx.strokeStyle = 'rgba(255,255,255,.75)'; octx.beginPath();
    pts.forEach((p, i) => (i ? octx.lineTo(p[0], p[1]) : octx.moveTo(p[0], p[1]))); octx.stroke();
    pts.forEach((p, i) => { octx.fillStyle = i === 0 ? '#3ad6ff' : `hsl(${40 + i * 3},100%,65%)`; octx.beginPath(); octx.arc(p[0], p[1], (i < 2 ? 4 : 2.4) * d, 0, 7); octx.fill(); });
    octx.fillStyle = '#fff';
    octx.fillText(n >= 600 ? 'הנקודה נשארת: היא בתוך הקבוצה (600 צעדים בלי לברוח)' : `הנקודה בורחת אחרי ${n + 1} צעדים`, 16 * d, H - 52 * d);
  }

  // ---------- dive video (map) ----------
  let recorder = null, chunks = [], cancelled = false, wake = null, resultURL = null;
  const pickMime = () => ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
  $('recBtn').onclick = () => {
    if (map.recording || map.flight || mode !== 'map') return;
    const mime = pickMime();
    if (!mime) { alert('הדפדפן אינו תומך בהקלטת וידאו מהקנבס.'); return; }
    const dest = { center: map.target.center, scale: map.target.scale }, dur = +$('dur').value * 1000;
    if ($('fromHome').checked) map.jump(HOME.center, HOME.scale);
    const [dx, dy] = bpSubD(dest.center, map.state.center).map(Math.abs);
    if (Math.abs(dest.scale - map.state.scale) < map.state.scale * 0.01 && dx < map.state.scale * 0.01 && dy < map.state.scale * 0.01) { alert('בחרו תחילה יעד: התקרבו למקום מעניין או בחרו מהרשימה, ואז הפעילו הקלטה.'); return; }
    chunks = []; cancelled = false;
    recorder = new MediaRecorder(canvas.captureStream(MOBILE ? 30 : 60), { mimeType: mime, videoBitsPerSecond: MOBILE ? 8e6 : 18e6 });
    if (navigator.wakeLock) navigator.wakeLock.request('screen').then((l) => { wake = l; }).catch(() => {});
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      map.recording = false; map.resScale = 1;
      if (wake) { wake.release().catch(() => {}); wake = null; }
      $('recOverlay').classList.remove('show'); $('recBtn').disabled = false;
      if (cancelled || !chunks.length) return;
      const blob = new Blob(chunks, { type: mime.split(';')[0] });
      if (resultURL) URL.revokeObjectURL(resultURL);
      resultURL = URL.createObjectURL(blob);
      $('resultVideo').src = resultURL; $('dlBtn').href = resultURL; $('dlBtn').download = 'mandelbrot-dive.' + (mime.startsWith('video/mp4') ? 'mp4' : 'webm');
      $('resultOverlay').classList.add('show');
    };
    map.recording = true;
    if (MOBILE) panel.classList.add('hidden');
    $('recBtn').disabled = true; $('recOverlay').classList.add('show'); $('recBar').firstElementChild.style.width = '0%';
    recorder.start(250);
    const t0 = performance.now();
    const prog = () => { if (!map.recording) return; $('recBar').firstElementChild.style.width = Math.min(100, (performance.now() - t0) / dur * 100) + '%'; requestAnimationFrame(prog); };
    prog();
    // a quiet moment first, then the dive, and a second of rest on arrival
    setTimeout(() => { if (map.recording) map.flyTo(dest.center, dest.scale, dur - 1600, () => setTimeout(() => { if (map.recording && recorder.state !== 'inactive') recorder.stop(); }, 1100)); }, 500);
  };
  $('recCancel').onclick = () => { if (recorder && recorder.state !== 'inactive') { cancelled = true; map.flight = null; recorder.stop(); } };
  $('closeResult').onclick = () => { $('resultOverlay').classList.remove('show'); $('resultVideo').pause(); };

  // ---------- go ----------
  resize();
  const H = new URLSearchParams(location.hash.slice(1));
  let start = null;
  if (H.get('x') && H.get('y') && +H.get('s') > 0) { const s = +H.get('s'); try { start = { center: bp(H.get('x'), H.get('y'), bitsFor(s)), scale: s }; } catch { start = null; } }
  if (H.get('p')) { const i = Math.min(PALETTES.length - 1, Math.max(0, H.get('p') | 0)); app.pal.set(i); document.querySelectorAll('.pal').forEach((x, k) => x.classList.toggle('active', k === i)); }
  const flyFirst = H.get('m') === 'fly' || Q.has('fly');
  if (start) map.jump(start.center, start.scale);
  else if (flyFirst) map.jump(HOME.center, HOME_SCALE);
  else { map.state.scale = HOME_SCALE * 3.2; map.flyTo(HOME.center, HOME_SCALE, 1800); }     // a small dive in, by way of greeting
  if (flyFirst) setTimeout(() => setMode('fly'), 50);

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (app.cycle) app.colOffset = (app.colOffset + dt * 0.04) % 1;
    if (switching) return;
    if (mode === 'map') map.update(now, dt); else if (fly) fly.update(now, dt);
  }
  requestAnimationFrame(frame);
  window.app = app; window.dbg = { map, get fly() { return fly; }, setMode, get mode() { return mode; }, PRESETS, go: (x, y, sc) => (mode === 'fly' ? fly : map).jump(bp(String(x), String(y), bitsFor(sc)), sc), preset: (i) => window.dbg.go(PRESETS[i].x, PRESETS[i].y, PRESETS[i].scale) };
}
try { boot(); } catch (e) { showErr(e); }
