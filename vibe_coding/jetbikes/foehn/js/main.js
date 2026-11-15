// FÖHN: boot, render loop, cameras, menus.
import * as THREE from '../lib/three.module.min.js';
import { initTerrain, groundAt, surfaceAt, T } from './sim/terrain.js';
import { buildScenery } from './sim/scenery.js';
import { Game } from './sim/game.js';
import { Autopilot } from './sim/autopilot.js';
import { Recorder } from './sim/replay.js';
import { planReplay } from './sim/director.js';
import { flights } from './ui/history.js';
import { StepFilm, LiveFilm, filmCaps, FPS } from './ui/film.js';
import { EVENTS, FREE, eventById } from './sim/courses.js';
import { len, yprOf, qrot, clamp, lerp } from './sim/vmath.js';
import { G } from './sim/bike.js';
import { U, TIMES, WEATHER, setTime, makeNoiseTexture, skyColor } from './render/env.js';
import { buildWeather } from './render/weather.js';
import { FEEL, FEEL_DEFAULT } from './sim/fcs.js';
import { buildTerrain, updateTerrain, bakeSunShadow } from './render/terrain.js';
import { buildSky } from './render/sky.js';
import { buildWater, renderReflection, water } from './render/water.js';
import { buildForest, updateForest } from './render/forest.js';
import { buildProps, Gates } from './render/props.js';
import { Targets } from './render/targets.js';
import { buildCastleMesh } from './render/castle.js';
import { hitCastle, nearCastle, indoor } from './sim/castle.js';
import { buildBike } from './render/bikeModel.js';
import { FX } from './render/fx.js';
import { Post } from './render/post.js';
import { Input } from './ui/input.js';
import { Audio, renderSound } from './ui/audio.js';
import { Hud, fmtTime } from './ui/hud.js';
import { t, evName, evTag, lang, setLang, langUrl } from './i18n.js';

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const load = (p, text) => { $('lbar').firstElementChild.style.width = Math.round(p * 100) + '%'; if (text) $('ltxt').textContent = text; return new Promise((r) => setTimeout(r, 0)); };

// ---------------- persistent state ----------------
const KEY = 'foehn.v1';
const save = (() => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } })();
save.best = save.best || {}; save.set = Object.assign({ time: 'afternoon', quality: 'high', invert: false, weather: 'clear', wind: 1 }, save.set || {});
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(save)); } catch { /* private mode */ } };
const b64 = (f32) => { const u = new Uint8Array(f32.buffer); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = (s) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };
const bestOf = (id) => { const b = save.best[id]; if (!b) return null; if (!b._g && b.ghost) { try { b._g = unb64(b.ghost); } catch { b._g = null; } } return { time: b.time, splits: b.splits, ghost: b._g, medal: b.medal }; };
if (Q.get('time') && TIMES[Q.get('time')]) save.set.time = Q.get('time');
if (Q.get('q')) save.set.quality = Q.get('q');
if (WEATHER[Q.get('wx')]) save.set.weather = Q.get('wx');
if (Q.has('wind')) save.set.wind = Math.max(0, Math.min(3, +Q.get('wind') | 0));
if (!WEATHER[save.set.weather]) save.set.weather = 'clear';

async function boot() {
  await load(0.03, t('load.valley'));
  const [meta, buf] = await Promise.all([fetch('assets/meta.json').then((r) => r.json()), fetch('assets/terrain.bin').then((r) => r.arrayBuffer())]);
  initTerrain(buf, meta);
  await load(0.12, t('load.trees'));
  buildScenery();
  await load(0.25, t('load.mountains'));

  // ---------------- renderer ----------------
  const canvas = $('c');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance', stencil: false });
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.NoToneMapping; renderer.autoClear = true; renderer.info.autoReset = true;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(64, 1, 0.3, 60000);
  const QUAL = { low: { scale: 0.7, msaa: 0, refl: false, shadow: 1024, forest: 0.6 }, medium: { scale: 0.85, msaa: 2, refl: true, shadow: 2048, forest: 0.8 }, high: { scale: 1, msaa: 4, refl: true, shadow: 2048, forest: 1 } };
  let qual = QUAL[save.set.quality] || QUAL.high;
  const post = new Post(renderer);
  let dynScale = 1, viewW = 1, viewH = 1;
  function resize() {
    viewW = innerWidth; viewH = innerHeight;
    const pr = Math.min(devicePixelRatio || 1, 1.5) * qual.scale * dynScale * (+Q.get('scale') || 1);
    renderer.setPixelRatio(1); renderer.setSize(Math.round(viewW * pr), Math.round(viewH * pr), false);
    post.msaa = qual.msaa; post.setSize(Math.round(viewW * pr), Math.round(viewH * pr));
    camera.aspect = viewW / viewH; camera.updateProjectionMatrix();
  }
  addEventListener('resize', resize);

  makeNoiseTexture();
  let sunInfo = setTime(save.set.time, renderer);
  const sun = new THREE.DirectionalLight(0xffffff, Math.PI); sun.castShadow = true;
  sun.shadow.mapSize.set(qual.shadow, qual.shadow); sun.shadow.camera.near = 1; sun.shadow.camera.far = 900;
  const SH = 70; Object.assign(sun.shadow.camera, { left: -SH, right: SH, top: SH, bottom: -SH }); sun.shadow.bias = -0.0006; sun.shadow.normalBias = 0.03;
  scene.add(sun, sun.target);
  const hemi = new THREE.HemisphereLight(0xffffff, 0xffffff, Math.PI * 0.75); scene.add(hemi);
  scene.fog = new THREE.FogExp2(0x8899aa, 0.00004);

  await buildTerrain(scene, (p) => load(0.25 + 0.3 * p));
  await load(0.56, t('load.shadows'));
  const sky = buildSky(scene);
  buildWater(scene, renderer);
  await load(0.62, t('load.forest'));
  buildForest(scene);
  await load(0.7, t('load.village'));
  const props = buildProps(scene);
  const castle = buildCastleMesh(scene);
  const gates = new Gates(scene), targets = new Targets(scene);
  const model = buildBike(); scene.add(model.root);
  const ghost = buildBike(true); ghost.root.visible = false; scene.add(ghost.root);
  const fx = new FX(scene, model);
  const weather = buildWeather(scene);
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null;

  function applyTime(key) {
    const wx = WEATHER[save.set.weather];
    sunInfo = setTime(key, renderer, wx);
    bakeSunShadow(sunInfo.sun);
    weather.set(save.set.weather); scene.fog.density = wx.fog2; rays = 1 - wx.ov;
    sun.color.copy(U.uSunCol.value); hemi.color.copy(U.uAmbSky.value); hemi.groundColor.copy(U.uAmbGnd.value);
    const hz = skyColor(U.uSunAz.value + 1.6, 0.03); scene.fog.color.setRGB(hz[0], hz[1], hz[2]);
    props.setNight(clamp(1 - sunInfo.sun[1] / 0.16, 0, 1)); castle.setNight(clamp(1 - sunInfo.sun[1] / 0.24, 0, 1));
    // reflections on the bike: the sky itself
    const es = new THREE.Scene(); const s2 = sky.clone(); es.add(s2);
    if (envRT) envRT.dispose();
    envRT = pmrem.fromScene(es, 0, 1, 70000); scene.environment = envRT.texture; scene.environmentIntensity = 0.55;
    exposure = ({ morning: 1.0, noon: 0.8, afternoon: 0.9, golden: 1.05, dusk: 1.6 }[key] || 1) * (wx.ov ? 1.25 : 1);
  }
  let exposure = 1, rays = 1;
  await load(0.78, t('load.sun'));
  applyTime(save.set.time);

  // ---------------- game ----------------
  const game = new Game(), input = new Input(), audio = new Audio(), hud = new Hud();
  input.invert = save.set.invert;
  const rec = new Recorder(game);
  let fcaps = null; filmCaps().then((c) => { fcaps = c; });      // how this browser can make a video file, if at all
  let rp = null;                          // the replay that is on screen now, if any
  let auto = null, mode = 'menu', paused = false, camMode = Q.get('cam') || 'chase', time = 0, flash = 0;
  const CAMS = ['chase', 'close', 'cockpit', 'cine'];
  const camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), cine = { p: new THREE.Vector3(), t: 0 };
  let camInit = false, fov = 64, lean = 0, lastCtl = { turn: 0, pitch: 0 };
  const touchUI = Q.has('touch') || matchMedia('(pointer: coarse)').matches;
  if (touchUI) input.enableTouch($('touch'));
  $('touch').classList.remove('on');

  function startEvent(id, demo = false) {
    game.windLevel = save.set.wind; game.windSeed = Q.has('seed') ? +Q.get('seed') : 1 + Math.floor(Math.random() * 1e6);   // no two runs get the same gusts
    if (mode === 'play') saveRun();                  // the run that is being left goes into the history
    game.load(id, demo ? null : bestOf(id));
    rec.reset(); run = demo ? 0 : Date.now(); runN = 0;
    auto = demo ? new Autopilot(game) : null;
    gates.setEvent(game.build); targets.setEvent(game); hud.setEvent(game);
    $('tFire').style.display = !demo && game.gunOn ? '' : 'none';
    mode = demo ? 'menu' : 'play'; paused = false; camInit = false; audio.streak = 0;
    if (!demo) { camMode = camMode === 'cine' ? 'chase' : camMode; hud.show(true); ['menu', 'pause', 'finish', 'help'].forEach((m) => $(m).classList.remove('on')); if (touchUI) $('touch').classList.add('on'); hintFor(id); }
    else { hud.show(false); game.stateT = 0.01; camMode = 'cine'; cine.t = 0; }
    ghost.root.visible = false;
  }
  function hintFor(id) {
    const first = !save.best[id];
    if (id === 'first' && first) hud.hint(t('hint.first'));
    else if (id === 'free') hud.hint(t('hint.free'));
    else if (id === 'castle' && first) hud.hint(t('hint.castle'));
    else if (id === 'hunt') hud.hint(t('hint.hunt'));
    else hud.hint('');
    setTimeout(() => hud.hint(''), 14000);
  }
  function toMenu() {
    startEvent('tour', true); buildCards();
    $('menu').classList.add('on'); ['pause', 'finish', 'help'].forEach((m) => $(m).classList.remove('on')); $('touch').classList.remove('on');
  }
  function buildCards() {
    const el = $('cards'); el.innerHTML = '';
    for (const ev of [...EVENTS, FREE]) {
      const b = save.best[ev.id], card = document.createElement('button'); card.className = 'card panel';
      const medals = ev.medals ? [1, 2, 3].map((k) => `<i class="medal ${b && b.medal >= k ? 'g' + k : ''}"></i>`).join('') : '';
      card.innerHTML = `<h3><span>${evName(ev)}</span><span>${medals}</span></h3><p>${evTag(ev)}</p><div class="meta"><span>${ev.medals ? t('card.gold') + '<span class="num">' + fmtTime(ev.medals[0]) + '</span>' : t('card.noClock')}</span><span>${b ? t('card.best') + '<span class="num">' + fmtTime(b.time) + '</span>' : ''}</span></div>`;
      card.onclick = () => { audio.start(); audio.click(); startEvent(ev.id); };
      el.appendChild(card);
    }
    $('bTime').textContent = t('set.time') + t('time.' + save.set.time);
    $('bWx').textContent = t('set.wx') + t('wx.' + save.set.weather); $('bWind').textContent = t('set.wind') + t('wind.' + save.set.wind);
    $('bQual').textContent = t('set.quality') + t('q.' + save.set.quality);
    $('bInv').textContent = t('set.axis') + t(save.set.invert ? 'axis.inv' : 'axis.norm');
    $('bLang').textContent = t('lang.other'); $('bShare').textContent = t('share');
  }
  $('bTime').onclick = () => { const k = Object.keys(TIMES); save.set.time = k[(k.indexOf(save.set.time) + 1) % k.length]; persist(); applyTime(save.set.time); buildCards(); };
  $('bWx').onclick = () => { const k = Object.keys(WEATHER); save.set.weather = k[(k.indexOf(save.set.weather) + 1) % k.length]; persist(); applyTime(save.set.time); buildCards(); };
  $('bWind').onclick = () => { save.set.wind = (save.set.wind + 1) % 4; persist(); game.setWind(save.set.wind); buildCards(); };
  $('bQual').onclick = () => { const k = ['low', 'medium', 'high']; save.set.quality = k[(k.indexOf(save.set.quality) + 1) % 3]; qual = QUAL[save.set.quality]; persist(); resize(); buildCards(); };
  $('bInv').onclick = () => { save.set.invert = !save.set.invert; input.invert = save.set.invert; persist(); buildCards(); };
  $('bLang').onclick = () => { setLang(lang === 'he' ? 'en' : 'he'); buildCards(); hud.setEvent(game); };
  $('bShare').onclick = async () => {
    const url = langUrl();
    if (navigator.share) { try { await navigator.share({ title: 'FÖHN', text: t('share.text'), url }); return; } catch (e) { if (e.name === 'AbortError') return; } }
    try { await navigator.clipboard.writeText(url); } catch { prompt('', url); return; }
    const b = $('bShare'); b.textContent = t('share.copied'); setTimeout(() => (b.textContent = t('share')), 1800);
  };
  $('bHelp').onclick = $('pHelp').onclick = () => $('help').classList.add('on');
  $('hClose').onclick = () => $('help').classList.remove('on');
  $('pResume').onclick = () => setPause(false);
  $('pReplay').onclick = $('fReplay').onclick = () => startReplay();
  $('rExit').onclick = () => stopReplay();
  $('rSave').onclick = () => startFilm();
  $('fFilm').onclick = () => startReplay({ video: true });
  $('bHist').onclick = () => openHistory();
  $('hiClose').onclick = () => $('hist').classList.remove('on');
  addEventListener('pagehide', () => saveRun());
  $('pRestart').onclick = $('fRetry').onclick = () => startEvent(game.ev.id);
  $('pMenu').onclick = $('fMenu').onclick = toMenu;
  $('fNext').onclick = () => { const i = EVENTS.findIndex((e) => e.id === game.ev.id); startEvent(i >= 0 && i < EVENTS.length - 1 ? EVENTS[i + 1].id : 'free'); };
  function setPause(p) { if (mode !== 'play') return; paused = p; $('pause').classList.toggle('on', p); if (audio.ctx) p ? audio.ctx.suspend() : audio.ctx.resume(); }
  addEventListener('visibilitychange', () => { if (document.hidden && mode === 'play' && game.state === 'fly') setPause(true); });

  const REC_EVENTS = ['go', 'gate', 'land', 'crash', 'respawn', 'shot', 'shotEnd', 'pop'];
  // what the gun does to eyes and ears, in the flight and again in its replay
  function gunFx(e) {
    const r = Math.random;
    if (e.type === 'shot') {
      audio.shot(); fx.shake = Math.max(fx.shake, 0.25);
      for (let i = 0; i < 10; i++) fx.fire.spawn(e.p, [e.v[0] * 0.06 + (r() - 0.5) * 5, e.v[1] * 0.06 + (r() - 0.5) * 5, e.v[2] * 0.06 + (r() - 0.5) * 5], [3, 2, 0.8], 0.9, 0.35 + 0.3 * r(), 0.09 + 0.07 * r(), 3, 3);
      const bv = game.bike.v;                                             // the puff of the muzzle leaves with the bike, or the rider flies through it
      for (let i = 0; i < 3; i++) fx.soft.spawn(e.p, [bv[0] + (e.v[0] - bv[0]) * 0.03 + (r() - 0.5) * 2, bv[1] + 0.5 + r(), bv[2] + (e.v[2] - bv[2]) * 0.03 + (r() - 0.5) * 2], [0.8, 0.8, 0.8], 0.16, 0.12 + 0.1 * r(), 0.35 + 0.25 * r(), 1.2, 2.5);
    } else if (e.type === 'pop') {
      audio.pop();
      for (let i = 0; i < 46; i++) { const a = r() * 6.283, u = r() * 2 - 1, q = Math.sqrt(1 - u * u), sp = 6 + 16 * r(); fx.soft.spawn(e.p, [Math.cos(a) * q * sp, u * sp, Math.sin(a) * q * sp], i % 3 ? [0.95, 0.25, 0.15] : [1, 0.85, 0.5], 0.9, 0.25 + 0.35 * r(), 0.5 + 0.6 * r(), 0.6, 2.2, 6); }
    } else if (e.type === 'shotEnd' && (e.end === 'ground' || e.end === 'water' || e.end === 'wall')) {
      const w = e.end === 'water', c = w ? [0.85, 0.9, 0.95] : [0.5, 0.44, 0.34];
      if (camera.position.distanceTo(tmpV.set(e.p[0], e.p[1], e.p[2])) < 400) audio.thud(w);
      for (let i = 0; i < 16; i++) fx.soft.spawn(e.p, [(r() - 0.5) * 6, 3 + 9 * r(), (r() - 0.5) * 6], c, 0.6, 0.4 + 0.5 * r(), 0.7 + 0.7 * r(), 2.5, 1.2, w ? 9 : 3);
    }
  }
  function onEvents() {
    for (const e of game.events) {
      if (mode !== 'play') { if (e.type === 'crash') fx.explode(e.p, e.v, e.water); if (e.type === 'respawn' && auto) auto.reset(); if (e.type === 'finish') setTimeout(() => mode === 'menu' && startEvent('tour', true), 2500); continue; }
      if (REC_EVENTS.includes(e.type)) rec.mark(e.type, e);
      switch (e.type) {
        case 'count': hud.message(String(e.n), '', 0.8); audio.count(false); break;
        case 'go': if (game.ev.medals) { hud.message(t('go'), '', 0.9); audio.count(true); } break;
        case 'gate': audio.gate(e.perfect, e.last); hud.delta(e.delta); if (e.perfect) hud.message('', t('perfect'), 0.7); flash = 0.06; break;
        case 'land': audio.gate(e.perfect, e.last); hud.delta(e.delta); if (!e.last) hud.message(t('landed'), t('landed.sub'), 1.4); break;
        case 'miss': audio.miss(); hud.message('', t('missed'), 2.2); break;
        case 'wrongway': audio.miss(); hud.message('', t('wrongway'), 2.2); break;
        case 'crash': fx.explode(e.p, e.v, e.water); audio.crash(e.water); hud.crash(e.reason); break;
        case 'warn': if (e.text === 'offpad') hud.message('', t('warn.offpad'), 1.2); else { hud.warn(t('warn.back')); audio.warn(); } break;
        case 'finish': onFinish(e); break;
        case 'shot': case 'pop': case 'shotEnd': gunFx(e); break;
        case 'shotMiss': hud.message('', t('gun.miss'), 0.9); break;
      }
    }
    game.events.length = 0;
  }
  function onFinish(r) {
    const id = game.ev.id, mine = run;
    setTimeout(() => run === mine && saveRun(), 7600);          // once the seconds after the line are on tape too
    if (r.isBest) { save.best[id] = { time: r.time, splits: r.splits, medal: Math.max(r.medal, (save.best[id] || {}).medal || 0), ghost: r.ghost.length < 60000 ? b64(r.ghost) : null, _g: r.ghost }; }
    else save.best[id].medal = Math.max(save.best[id].medal || 0, r.medal);
    try { const c = { ...save, best: Object.fromEntries(Object.entries(save.best).map(([k, v]) => [k, { time: v.time, splits: v.splits, medal: v.medal, ghost: v.ghost }])) }; localStorage.setItem(KEY, JSON.stringify(c)); } catch { /* quota */ }
    $('fname').textContent = evName(game.ev); $('ftime').textContent = fmtTime(r.time);
    $('fmedal').innerHTML = r.medal ? `<i class="medal g${r.medal}" style="width:20px;height:20px"></i> ${t('medal.' + r.medal)}` : t('medal.none');
    $('fbest').innerHTML = r.isBest ? (r.prev ? `${t('fin.newBest')}<span class="num">${fmtTime(r.prev)}</span>` : t('fin.firstBest')) : `${t('fin.best')}<span class="num">${fmtTime(r.prev)}</span>`;
    const m = game.ev.medals, nextT = r.medal < 3 ? m[2 - r.medal] : null;
    $('fstat').innerHTML = (game.ev.gun ? `${t('fin.shots')}${r.shots} · ${t('fin.crashes')}${r.crashes}` : `${t('fin.crashes')}${r.crashes} · ${t('fin.misses')}${r.misses}`) + (nextT ? ` · ${t('fin.next')}<span class="num">${fmtTime(nextT)}</span>` : '');
    setTimeout(() => { if (mode === 'play' && game.state === 'finished') { $('finish').classList.add('on'); $('touch').classList.remove('on'); } }, 1800);
  }

  // ---------------- replay ----------------
  // The run that was just flown, cut like a film (sim/director.js). Nothing is simulated again:
  // the recorded state is written into the bike frame by frame, and put back when it is over.
  function startReplay(opts = {}) {
    if (rp || rec.duration < 2 || (mode !== 'play' && !opts.library)) return;
    if (opts.video && !fcaps) { alert(t('v.unsupported')); return; }
    rp = { saved: rec.snapshot(), wasPaused: paused, lean, plan: planReplay(rec, game.cps, game.bike.mp.cg), t: rec.t0, hold: false, o: {}, ending: false, sec: -1, library: !!opts.library, meta: opts.meta || runMeta(), film: null, popped: new Set(), flying: [],
      shots: rec.events.filter((e) => e.type === 'shot').map((e) => ({ t: e.t, p: e.p, v: e.v, end: (rec.events.find((x) => x.type === 'shotEnd' && x.id === e.id && x.t >= e.t) || { t: e.t + 4 }).t })) };
    mode = 'replay'; paused = false; if (audio.ctx) audio.ctx.resume();
    ['pause', 'finish', 'help', 'touch'].forEach((m) => $(m).classList.remove('on')); hud.show(false); ghost.root.visible = false;
    $('replay').classList.add('on'); $('rfade').classList.add('on');
    requestAnimationFrame(() => requestAnimationFrame(() => $('rfade').classList.remove('on')));      // fade in from black
    if (opts.video) startFilm();
  }
  function stopReplay() {
    if (!rp) return;
    if (rp.film) rp.film.cancel();
    audio.tape = null;
    rec.restore(rp.saved); lean = rp.lean;
    const wasPaused = rp.wasPaused, library = rp.library; rp = null; mode = 'play'; camInit = false;
    $('replay').classList.remove('on'); $('rfade').classList.remove('on'); $('replay').classList.remove('hold', 'film');
    if (library) {                                   // a flight from the history: back to the list, and the day as it was set
      save.set.weather = libEnv.weather; applyTime(libEnv.time);
      mode = 'menu'; toMenu(); openHistory(); return;
    }
    hud.show(true);
    if (game.state === 'finished') $('finish').classList.add('on'); else if (wasPaused) setPause(true); else if (touchUI) $('touch').classList.add('on');
  }
  function replaySeek(t) { rp.t = clamp(t, rec.t0, rec.t1 - 0.5); lean = rec.apply(rp.t); }
  // one frame of the replay; returns the time that passed in the recording (less than dt in slow motion)
  function replayStep(dt) {
    const sdt = dt * rp.plan.rate(rp.t), was = rp.t;
    rp.t = Math.min(rp.t + sdt, rec.t1);
    lean = rec.apply(rp.t);
    for (const e of rec.events) {
      if (e.t <= was || e.t > rp.t) continue;
      if (e.type === 'crash') { fx.explode(e.p, e.v, e.water); audio.crash(e.water); }
      else if (e.type === 'gate' || e.type === 'land') { audio.gate(e.perfect, e.last); flash = 0.06; }
      else if (e.type === 'shot' || e.type === 'pop' || e.type === 'shotEnd') gunFx(e);
    }
    // balloons already shot at this moment of the film, and the rounds in the air (their flight is plain ballistics)
    rp.popped.clear(); for (const e of rec.events) if (e.type === 'pop' && e.t <= rp.t) rp.popped.add(e.k);
    rp.flying.length = 0;
    for (const sh of rp.shots) { const u = rp.t - sh.t; if (u >= 0 && rp.t < sh.end) rp.flying.push({ p: [sh.p[0] + sh.v[0] * u, sh.p[1] + sh.v[1] * u - 0.5 * G * u * u, sh.p[2] + sh.v[2] * u], v: [sh.v[0], sh.v[1] - G * u, sh.v[2]] }); }
    if (rp.t >= rec.t1 && !rp.ending && rp.film && rp.film.step) { rp.ending = true; rp.last = true; }      // the film ends on this very frame
    else if (rp.t >= rec.t1 && !rp.ending) { const me = rp; rp.ending = true; $('rfade').classList.add('on'); setTimeout(() => rp === me && (me.film ? endFilm() : stopReplay()), 650); }
    const sec = Math.floor(rp.t - rec.t0);
    if (sec !== rp.sec) { rp.sec = sec; $('rtime').textContent = fmtTime(sec).replace(/\.\d+$/, ''); }
    $('rprog').style.width = (100 * (rp.t - rec.t0) / Math.max(1, rec.duration)).toFixed(2) + '%';
    return sdt;
  }
  // ---- the replay as a video file ----
  function startFilm() {
    if (!rp || rp.film) return;
    if (!fcaps) { alert(t('v.unsupported')); return; }
    audio.start();
    rp.t = rec.t0; rp.ending = false; rp.hold = false; rp.sec = -1; lean = rec.apply(rp.t);
    $('rfade').classList.remove('on'); $('replay').classList.remove('hold'); $('replay').classList.add('film');
    const m = rp.meta, d = new Date(m.id), z = (n) => String(n).padStart(2, '0');
    const cap = { text: flightTitle(m) + ' · ' + d.toLocaleDateString(lang === 'he' ? 'he-IL' : 'en-GB'), rtl: lang === 'he' };
    $('rrec').innerHTML = t('r.rec');
    if (fcaps.step) {
      // made frame by frame (ui/film.js): count the frames first, slow motion and all
      let n = 0; for (let u = rec.t0; u < rec.t1; u += Math.max(0.05, rp.plan.rate(u)) / FPS) n++;
      rp.film = new StepFilm(canvas, cap, fcaps.step, n / FPS);
      audio.tape = []; audio.tapeT = 0; audio.streak = 0;
    } else rp.film = new LiveFilm(canvas, cap, fcaps.live, audio.tap());
    rp.name = `foehn-${m.ev}-${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}.${rp.film.ext}`;
  }
  async function endFilm() {
    const me = rp, film = rp.film, name = rp.name, tape = audio.tape; audio.tape = null;
    rp.wrap = true; rp.last = false; $('rrec').innerHTML = t('r.wrap'); $('rfade').classList.add('on');
    let blob = null;
    try { blob = film.step ? await film.finish(await renderSound(tape, film.time)) : await film.finish(); } catch (e) { console.error(e); if (rp === me) alert(t('v.failed')); }
    if (rp !== me) return;                           // cancelled while the file was being closed
    rp.film = null; stopReplay();
    if (blob && blob.size) showFilm(blob, name);
  }
  let filmUrl = null;
  function showFilm(blob, name) {
    if (filmUrl) URL.revokeObjectURL(filmUrl);
    filmUrl = URL.createObjectURL(blob);
    const a = $('vSave'); a.href = filmUrl; a.download = name;
    $('vNote').textContent = `${name} · ${(blob.size / 1048576).toFixed(1)} MB`;
    const file = new File([blob], name, { type: blob.type }), can = !!(navigator.canShare && navigator.canShare({ files: [file] }));
    $('vShare').style.display = can ? '' : 'none';
    $('vShare').onclick = () => navigator.share({ files: [file], title: 'FÖHN', text: t('share.text') }).catch(() => {});
    $('vid').classList.add('on');
    a.click();                                        // straight into the downloads folder; the button is there to do it again
  }
  $('vClose').onclick = () => { $('vid').classList.remove('on'); if (filmUrl) { URL.revokeObjectURL(filmUrl); filmUrl = null; } };

  // ---- the history of flights ----
  // Every run of ten seconds or more is kept in this browser (ui/history.js), finished or not.
  let run = 0, runN = 0, libEnv = null;      // the id (start time) of the run being flown, and how much of it is saved
  const runMeta = () => {
    const r = (rp ? rp.saved.state : game.state) === 'finished' ? game.result : null;
    return { id: run || Date.now(), ev: game.ev.id, done: !!r, time: r ? r.time : null, medal: r ? r.medal : 0, crashes: game.crashes, misses: game.misses, dur: rec.duration, wx: save.set.weather, tod: save.set.time, wind: save.set.wind, score: Math.round(game.score) };
  };
  function saveRun() {
    const cnt = rec.n - rec.base;
    if (!run || cnt === runN || rec.duration < 10) return;
    runN = cnt;
    flights.put(runMeta(), rec.dump()).catch((e) => console.warn('flight not saved', e));
  }
  const flightTitle = (m) => { const ev = eventById(m.ev); return (ev ? evName(ev) : m.ev) + (m.done && m.time != null ? ' · ' + fmtTime(m.time) : ''); };
  async function openHistory() {
    const el = $('hlist'); $('hist').classList.add('on');
    let list = [];
    try { list = await flights.list(); } catch { /* no storage here */ }
    if (!list.length) { el.innerHTML = `<p style="opacity:.75">${t('hi.empty')}</p>`; return; }
    el.innerHTML = '';
    for (const m of list) {
      const row = document.createElement('div'); row.className = 'hrow';
      const when = new Date(m.id).toLocaleString(lang === 'he' ? 'he-IL' : 'en-GB', { dateStyle: 'short', timeStyle: 'short' });
      const medal = m.medal ? ` <i class="medal g${m.medal}"></i>` : '', mm = Math.floor(m.dur / 60), ss = String(Math.floor(m.dur % 60)).padStart(2, '0');
      row.innerHTML = `<div><b>${flightTitle(m)}</b>${medal}${m.done || m.ev === 'free' ? '' : ` <span class="und">${t('hi.undone')}</span>`}<br><small><span class="num">${when}</span> · <span class="num">${mm}:${ss}</span> · ${t('wx.' + m.wx)} · ${t('set.wind')}${t('wind.' + m.wind)}</small></div>`
        + `<div class="hb"><button class="b" data-a="w">${t('hi.watch')}</button><button class="b" data-a="v">${t('hi.film')}</button><button class="b" data-a="d" aria-label="delete">✕</button></div>`;
      row.querySelector('[data-a=w]').onclick = () => watchFlight(m, false);
      row.querySelector('[data-a=v]').onclick = () => watchFlight(m, true);
      row.querySelector('[data-a=d]').onclick = async () => { if (!confirm(t('hi.delq'))) return; await flights.del(m.id).catch(() => {}); openHistory(); };
      el.appendChild(row);
    }
  }
  async function watchFlight(m, video) {
    if (rp || mode !== 'menu') return;
    if (video && !fcaps) { alert(t('v.unsupported')); return; }
    audio.start();
    let d = null; try { d = await flights.get(m.id); } catch { /* gone */ }
    if (!d || !eventById(m.ev) || rp || mode !== 'menu') return;
    run = 0;
    game.load(m.ev, null); game.events.length = 0; auto = null; gates.setEvent(game.build); targets.setEvent(game);
    if (!rec.load(d)) { toMenu(); return; }
    libEnv = { weather: save.set.weather, time: save.set.time };            // the film gets the weather and the hour it was flown in
    if (m.wx !== save.set.weather || m.tod !== save.set.time) { save.set.weather = WEATHER[m.wx] ? m.wx : 'clear'; applyTime(TIMES[m.tod] ? m.tod : save.set.time); }
    ['menu', 'hist'].forEach((x) => $(x).classList.remove('on'));
    startReplay({ library: true, video, meta: m });
  }

  function replayCamera() {
    const b = game.bike, o = rp.o; o.on = false;
    rp.plan.shotAt(rp.t).cam(rp.t, o);
    camera.position.set(o.p[0], o.p[1], o.p[2]); camera.up.set(o.up[0], o.up[1], o.up[2]); camera.lookAt(o.l[0], o.l[1], o.l[2]);
    // a camera bolted to the bike shakes with it; one on the ground shakes when the bike thunders past
    const d = Math.hypot(o.p[0] - b.p[0], o.p[1] - b.p[1], o.p[2] - b.p[2]), alive = game.state !== 'crashed';
    const shake = fx.shake * 0.5 + (!alive ? 0 : o.on ? b.abOut * 0.03 + Math.max(0, len(b.v) - 80) * 0.0004 : Math.min(0.1, b.thrust / 14000 * 60 / (d * d + 60) * 0.12));
    if (shake > 0.001) { camera.position.x += (Math.random() - 0.5) * shake; camera.position.y += (Math.random() - 0.5) * shake; camera.position.z += (Math.random() - 0.5) * shake; }
    fov = o.fov; camera.fov = fov; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  }

  // ---------------- cameras ----------------
  const tmpV = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), tmpV2 = new THREE.Vector2();
  camera.worldToScreen = (x, y, z) => { tmpV.set(x, y, z).applyMatrix4(camera.matrixWorldInverse); const behind = tmpV.z > 0; tmpV.applyMatrix4(camera.projectionMatrix); return { x: tmpV.x, y: tmpV.y, behind }; };
  function updateCamera(dt) {
    if (rp) return replayCamera();
    const b = game.bike, V = len(b.v), [yaw] = yprOf(b.q);
    const fw = qrot(b.q, [1, 0, 0]), up = qrot(b.q, [0, 1, 0]);
    // direction of travel, blending from heading (hover) to velocity (flight)
    const k = clamp((V - 6) / 20, 0, 1), fh = [Math.cos(yaw), 0, -Math.sin(yaw)];
    let dir = [lerp(fh[0], b.v[0] / (V || 1), k), lerp(0, b.v[1] / (V || 1), k) * 0.6, lerp(fh[2], b.v[2] / (V || 1), k)];
    const dl = Math.hypot(dir[0], dir[1], dir[2]) || 1; dir = dir.map((v) => v / dl);
    // the camera feels the after-burner, the speed, and the ground rushing by close underneath
    const rush = clamp((V - 35) / 60, 0, 1) * clamp(1 - b.agl / 14, 0, 1);
    const shake = fx.shake * 0.5 + b.abOut * 0.045 + Math.max(0, V - 80) * 0.0006 + rush * 0.02;
    let targetFov = 56 + Math.min(15, V * 0.15) + b.abOut * 7 + rush * 3;
    if (camMode === 'free') { /* placed by hand (debug, screenshots) */ } else if (camMode === 'cockpit') {
      const p = b.toWorld([0.16, 1.5, 0]);
      camera.position.set(p[0], p[1], p[2]);
      tmpQ.set(b.q[1], b.q[2], b.q[3], b.q[0]);
      camera.quaternion.copy(tmpQ).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, 0))).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.10, -lean * 0.25, 0)));
      targetFov += 12;
    } else if (camMode === 'cine') {
      cine.t -= dt;
      const d = cine.p.distanceTo(tmpV.set(b.p[0], b.p[1], b.p[2]));
      if (cine.t <= 0 || d > 560 || !camInit) {
        // a new vantage point ahead of the bike, beside its path, low over the ground
        // (never further ahead than the bike can be followed from: at 450 km/h a point 3 s ahead was already out of range, and the view jumped every frame)
        const side = Math.random() < 0.5 ? -1 : 1, ahead = Math.min(2.2 + Math.random() * 1.5, 300 / Math.max(V, 1));
        const x = b.p[0] + b.v[0] * ahead + fw[2] * side * (14 + Math.random() * 40) + fh[0] * 40, z = b.p[2] + b.v[2] * ahead - fw[0] * side * (14 + Math.random() * 40) + fh[2] * 40;
        const y = Math.max(surfaceAt(x, z) + 2.5 + Math.random() * 10, b.p[1] + b.v[1] * ahead - 12 + Math.random() * 16);
        cine.p.set(x, y, z); cine.t = 5 + Math.random() * 3;
        if (hitCastle([x, y, z], 2)) cine.t = 0;                 // inside a wall: pick again next frame
        if (!camInit) camLook.copy(tmpV); camInit = true;
      }
      camera.position.copy(cine.p); camLook.lerp(tmpV, 1 - Math.exp(-dt * 9)); camera.up.set(0, 1, 0); camera.lookAt(camLook);
      targetFov = clamp(2200 / (d + 30), 16, 55);
    } else {
      const dist = camMode === 'close' ? 3.1 : 4.8 + V * 0.008, h = camMode === 'close' ? 0.95 : 1.5;
      tmpV.set(b.p[0] - dir[0] * dist, b.p[1] - dir[1] * dist + h, b.p[2] - dir[2] * dist);
      if (!camInit) { camPos.copy(tmpV); camInit = true; }
      camPos.lerp(tmpV, 1 - Math.exp(-dt * (5 + V * 0.06)));
      // stay close enough at speed, and above the ground
      const dx = camPos.x - b.p[0], dy = camPos.y - b.p[1], dz = camPos.z - b.p[2], dd = Math.hypot(dx, dy, dz), mx = dist + 2.5;
      if (dd > mx) camPos.set(b.p[0] + dx / dd * mx, b.p[1] + dy / dd * mx, b.p[2] + dz / dd * mx);
      camPos.y = Math.max(camPos.y, surfaceAt(camPos.x, camPos.z) + 0.9);
      if (nearCastle(b.p)) {
        // between walls: never let masonry come between the camera and the bike
        const ax = b.p[0], ay = b.p[1] + 0.9, az = b.p[2];
        for (let k = 1; k <= 12; k++) {
          const u = k / 12;
          if (hitCastle([ax + (camPos.x - ax) * u, ay + (camPos.y - ay) * u, az + (camPos.z - az) * u], 0.45)) { const f = Math.max(0.12, (k - 1) / 12); camPos.set(ax + (camPos.x - ax) * f, ay + (camPos.y - ay) * f, az + (camPos.z - az) * f); break; }
        }
      }
      camera.position.copy(camPos);
      camLook.set(b.p[0] + dir[0] * 9, b.p[1] + dir[1] * 9 + 1.0, b.p[2] + dir[2] * 9);
      // bank a little with the bike
      const roll = yprOf(b.q)[2];
      camera.up.set(Math.sin(roll * 0.28) * fh[2] * -1, 1, Math.sin(roll * 0.28) * fh[0]).normalize();
      camera.lookAt(camLook);
    }
    if (shake > 0.001) { camera.position.x += (Math.random() - 0.5) * shake; camera.position.y += (Math.random() - 0.5) * shake; camera.position.z += (Math.random() - 0.5) * shake; }
    if (camMode !== 'free') fov += (targetFov - fov) * Math.min(1, dt * 3);
    camera.fov = fov; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  }

  // ---------------- frame ----------------
  resize();
  let last = performance.now(), fpsAcc = 0, fpsN = 0;
  const hideInRefl = [model.root, ghost.root, fx.soft.pts, fx.fire.pts, gates.group, gates.beam, ...weather.objects];
  function frame(now) {
    requestAnimationFrame(frame);
    let dt = Math.min(0.05, (now - last) / 1000); last = now;
    // keys
    for (const k of input.pressed) {
      if (k === 'KeyM') { audio.setMuted(!audio.muted); }
      if (mode === 'replay') {
        if (k === 'Escape' || ((k === 'Enter' || k === 'KeyV' || k === 'PadA') && !rp.film)) stopReplay();
        else if (rp.film) { /* nothing else while the film is being recorded */ }
        else if (k === 'KeyS') startFilm();
        else if (k === 'Space' || k === 'KeyP') { rp.hold = !rp.hold; $('replay').classList.toggle('hold', rp.hold); }
        else if (k === 'ArrowRight') replaySeek(rp.t + 5);
        else if (k === 'ArrowLeft') replaySeek(rp.t - 5);
        else if (k === 'KeyC') { const sh = rp.plan.shotAt(rp.t); if (sh.b < rec.t1 - 0.6) replaySeek(sh.b + 0.01); }
      } else if (mode === 'play') {
        if (k === 'KeyV' && (paused || game.state === 'finished')) { startReplay(); continue; }
        if (k === 'Escape' || k === 'KeyP') { if ($('help').classList.contains('on')) $('help').classList.remove('on'); else setPause(!paused); }
        if (k === 'KeyR') startEvent(game.ev.id);
        if (k === 'KeyC') { camMode = CAMS[(CAMS.indexOf(camMode) + 1) % CAMS.length]; camInit = false; }
        if (k === 'KeyH') $('help').classList.toggle('on');
        if ((k === 'Enter' || k === 'PadA') && game.state === 'finished') startEvent(game.ev.id);
      }
    }
    input.pressed.length = 0;
    const stepping = rp && rp.film && rp.film.step && !rp.wrap;     // a film is being made: every frame is 1/60 s of it, however long it took
    if (stepping) { dt = 1 / FPS; audio.tapeT = rp.film.time; }
    if (!paused && !(rp && rp.hold) && !(stepping && rp.film.full)) {
      const realDt = dt;
      let ctl = lastCtl;
      if (rp) dt = replayStep(realDt);
      else {
        ctl = auto ? auto.update() : input.read();
        lastCtl = ctl;
        game.update(dt, ctl);
        if (mode === 'play') rec.push(dt, lean);
        onEvents();
        lean += ((ctl.turn || 0) - lean) * Math.min(1, dt * 5);
      }
      time += dt; U.uTime.value = time;
      const b = game.bike, alive = game.state !== 'crashed';
      model.root.visible = alive; model.update(b, time, lean, dt);
      const gp = mode === 'play' && game.state === 'fly' ? game.ghostAt(game.total) : null;
      ghost.root.visible = !!gp; if (gp) ghost.setPose(gp);
      fx.update(dt, b, game.sim.windW, alive);
      water.wash.set(fx.washPos[0], fx.washPos[1], 6, fx.washStrength);
      targets.update(time, rp ? (k) => !rp.popped.has(k) : (k) => !game.cps0[k].popped, rp || game.state === 'finished' ? null : game.nextCp, rp ? rp.flying : game.gun.shots);
      gates.update(game, time); gates.matDisc.visible = !rp; if (rp) gates.beam.visible = false;      // the beacon and the glass of the next ring are for the rider, not for the film
      castle.update(time, game.sim.windW, camera.position, renderer.getDrawingBufferSize(tmpV2).y, camera.fov);
      audio.engine(b, alive && mode !== 'menu', camera.position.distanceTo(model.root.position), rp ? !!rp.o.on : camMode === 'cockpit');
      if (mode === 'play') {
        if (b.abLock && Math.floor(time * 4) % 2 === 0 && ctl.boost) hud.warn(t('warn.heat'));
        else if (b.stall > 0.6 && b.airspeed > 30) hud.warn(t('warn.stall'));
        else if (b.fuel < 8 && Math.floor(time * 2) % 2 === 0) hud.warn(t('warn.fuel'));
      }
      updateCamera(dt);
      dt = realDt;
      const roof = indoor([camera.position.x, camera.position.y, camera.position.z]), wNow = game.sim.windW;
      weather.update(dt, camera, wNow, roof, renderer.getDrawingBufferSize(tmpV2).y);
      audio.weather(save.set.weather === 'rain' ? 1 - 0.7 * roof : 0, Math.hypot(wNow[0], wNow[1], wNow[2]) * (1 - 0.6 * roof), mode !== 'menu');
    }
    // ---- render ----
    const b = game.bike;
    updateTerrain(camera); updateForest(camera, qual.forest);
    sky.position.copy(camera.position); sky.updateMatrix();
    // shadow frustum follows the bike
    const film = !!rp || camMode === 'cine', s = sunInfo.sun, fx0 = b.p[0], fy0 = b.p[1], fz0 = b.p[2];
    sun.target.position.set(fx0, fy0, fz0); sun.position.set(fx0 + s[0] * 400, fy0 + s[1] * 400, fz0 + s[2] * 400); sun.target.updateMatrixWorld();
    sun.intensity = s[1] > 0.02 ? Math.PI : 0;
    if (qual.refl) renderReflection(renderer, scene, camera, hideInRefl, (v) => { updateTerrain(v, 2.5); });
    else water.lakes[0].material.uniforms.uHasRefl.value = 0;
    updateTerrain(camera);
    renderer.setRenderTarget(post.main); renderer.clear(); renderer.render(scene, camera);
    if (sun.shadow.map) { U.uShadowMap.value = sun.shadow.map.texture; U.uShadowMat.value.copy(sun.shadow.matrix); U.uShadowOn.value = 1; }
    flash = Math.max(0, flash - dt * 0.5);
    post.finish(camera, s, { blur: film ? 0.3 : 0.5, speed: clamp((b.airspeed - 35) / 80, 0, 1) * (film ? 0 : 1) + b.abOut * (film ? 0 : 0.3), time, exposure, flash: Math.min(0.6, flash + Math.max(0, fx.flashT - 0.42) * 6), rays });
    if (rp && rp.film && !rp.wrap && !(stepping && rp.film.full)) { rp.film.grab(Math.max(1 - (rp.t - rec.t0) / 0.5, 1 - (rec.t1 - rp.t) / 0.6, 0)); if (rp.last) endFilm(); }
    if (!paused && !rp) hud.update(dt, game, camera, viewW, viewH);
    // adaptive resolution
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 1.5) {
      const fps = fpsN / fpsAcc; game.fps = fps; fpsAcc = 0; fpsN = 0;
      if (!Q.has('fixed')) {
        if (fps < 44 && dynScale > 0.6) { dynScale = Math.max(0.6, dynScale - 0.1); resize(); } else if (fps > 58 && dynScale < 1) { dynScale = Math.min(1, dynScale + 0.05); resize(); }
      }
    }
  }

  // ---------------- ?tune: the numbers of the ride, as sliders ----------------
  if (Q.has('tune')) {
    const TKEY = 'foehn.feel';
    try { Object.assign(FEEL, JSON.parse(localStorage.getItem(TKEY)) || {}); } catch { /* nothing saved */ }
    const rows = [['stick', 'עיכוב המוט (שנ\')', 0.02, 0.2, 0.01], ['rollRate', 'קצב גלגול', 1.5, 6, 0.1], ['rollAcc', 'תאוצת גלגול', 5, 30, 1], ['yawRate', 'קצב סבסוב', 1, 3.5, 0.1], ['yawAcc', 'תאוצת סבסוב', 2, 14, 0.5],
      ['bank', 'הטיה מרבית (רדיאן)', 1.0, 1.33, 0.01], ['kick', 'בעיטת צד (מ/שנ²)', 0, 20, 1], ['strafe', 'הזזה בריחוף (מ/שנ\')', 5, 14, 1], ['gust', 'בלימת משבים', 0, 1, 0.05], ['hairpin', 'פניית סיכה אוטומטית', 0, 1, 1], ['recoil', 'רתע הירייה (מ/שנ\')', 0, 4, 0.1], ['muzzle', 'מהירות הכדור (מ/שנ\')', 120, 600, 10]];
    const el = document.createElement('div'); el.id = 'tune'; el.className = 'panel'; document.body.appendChild(el);
    const draw = () => {
      el.innerHTML = '<b>תחושת הרכיבה</b>' + rows.map(([k, name, a, b, st]) => `<label><span>${name}</span><span class="num" id="tv_${k}">${FEEL[k]}</span></label><input type="range" data-k="${k}" min="${a}" max="${b}" step="${st}" value="${FEEL[k]}">`).join('') + '<div class="rowbtn"><button class="b" id="tReset">איפוס</button><button class="b" id="tCopy">העתקה</button></div>';
      for (const i of el.querySelectorAll('input')) i.oninput = () => { FEEL[i.dataset.k] = +i.value; $('tv_' + i.dataset.k).textContent = i.value; try { localStorage.setItem(TKEY, JSON.stringify(FEEL)); } catch { /* private mode */ } };
      $('tReset').onclick = () => { Object.assign(FEEL, FEEL_DEFAULT); try { localStorage.removeItem(TKEY); } catch { /* private mode */ } draw(); };
      $('tCopy').onclick = async () => { const s = JSON.stringify(FEEL); try { await navigator.clipboard.writeText(s); $('tCopy').textContent = 'הועתק'; } catch { prompt('', s); } };
    };
    draw();
    el.addEventListener('pointerup', () => document.activeElement && document.activeElement.blur());   // or the arrow keys would move the slider instead of the bike
  }

  // ---------------- go ----------------
  await load(0.9, t('load.engines'));
  toMenu();
  if (Q.get('event') && [...EVENTS, FREE].some((e) => e.id === Q.get('event'))) { startEvent(Q.get('event')); if (Q.has('demo')) { auto = new Autopilot(game); game.stateT = 0.01; } }
  frame(performance.now());
  await load(1, '');
  $('load').classList.add('off');
  addEventListener('pointerdown', () => audio.start(), { once: true }); addEventListener('keydown', () => audio.start(), { once: true });
  window.game = game; window.dbg = { scene, camera, renderer, post, setCam: (m) => { camMode = m; camInit = false; }, free: (x, y, z, lx, ly, lz, f = 55) => { camMode = 'free'; camera.position.set(x, y, z); camera.up.set(0, 1, 0); camera.lookAt(lx, ly, lz); fov = f; }, hud: (on) => hud.show(on), applyTime, get mode() { return mode; }, startEvent, startReplay, stopReplay, rec, flights, openHistory, get rp() { return rp; }, U, sun };
}
if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
boot().catch((e) => { const el = $('err'); el.style.display = 'block'; el.textContent += (e.stack || e) + '\n'; console.error(e); });
