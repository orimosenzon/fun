// FÖHN: boot, render loop, cameras, menus.
import * as THREE from '../lib/three.module.min.js';
import { initTerrain, groundAt, surfaceAt, T } from './sim/terrain.js';
import { buildScenery } from './sim/scenery.js';
import { Game } from './sim/game.js';
import { Autopilot } from './sim/autopilot.js';
import { EVENTS, FREE } from './sim/courses.js';
import { len, yprOf, qrot, clamp, lerp } from './sim/vmath.js';
import { G } from './sim/bike.js';
import { U, TIMES, setTime, makeNoiseTexture, skyColor } from './render/env.js';
import { buildTerrain, updateTerrain, bakeSunShadow } from './render/terrain.js';
import { buildSky } from './render/sky.js';
import { buildWater, renderReflection, water } from './render/water.js';
import { buildForest, updateForest } from './render/forest.js';
import { buildProps, Gates } from './render/props.js';
import { buildBike } from './render/bikeModel.js';
import { FX } from './render/fx.js';
import { Post } from './render/post.js';
import { Input } from './ui/input.js';
import { Audio } from './ui/audio.js';
import { Hud, fmtTime } from './ui/hud.js';
import { t, evName, evTag, lang, setLang, langUrl } from './i18n.js';

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const load = (p, text) => { $('lbar').firstElementChild.style.width = Math.round(p * 100) + '%'; if (text) $('ltxt').textContent = text; return new Promise((r) => setTimeout(r, 0)); };

// ---------------- persistent state ----------------
const KEY = 'foehn.v1';
const save = (() => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } })();
save.best = save.best || {}; save.set = Object.assign({ time: 'afternoon', quality: 'high', invert: false }, save.set || {});
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(save)); } catch { /* private mode */ } };
const b64 = (f32) => { const u = new Uint8Array(f32.buffer); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = (s) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };
const bestOf = (id) => { const b = save.best[id]; if (!b) return null; if (!b._g && b.ghost) { try { b._g = unb64(b.ghost); } catch { b._g = null; } } return { time: b.time, splits: b.splits, ghost: b._g, medal: b.medal }; };
if (Q.get('time') && TIMES[Q.get('time')]) save.set.time = Q.get('time');
if (Q.get('q')) save.set.quality = Q.get('q');

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
  const gates = new Gates(scene);
  const model = buildBike(); scene.add(model.root);
  const ghost = buildBike(true); ghost.root.visible = false; scene.add(ghost.root);
  const fx = new FX(scene, model);
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null;

  function applyTime(key) {
    sunInfo = setTime(key, renderer);
    bakeSunShadow(sunInfo.sun);
    sun.color.copy(U.uSunCol.value); hemi.color.copy(U.uAmbSky.value); hemi.groundColor.copy(U.uAmbGnd.value);
    const hz = skyColor(U.uSunAz.value + 1.6, 0.03); scene.fog.color.setRGB(hz[0], hz[1], hz[2]);
    props.setNight(clamp(1 - sunInfo.sun[1] / 0.16, 0, 1));
    // reflections on the bike: the sky itself
    const es = new THREE.Scene(); const s2 = sky.clone(); es.add(s2);
    if (envRT) envRT.dispose();
    envRT = pmrem.fromScene(es, 0, 1, 70000); scene.environment = envRT.texture; scene.environmentIntensity = 0.55;
    exposure = { morning: 1.0, noon: 0.8, afternoon: 0.9, golden: 1.05, dusk: 1.6 }[key] || 1;
  }
  let exposure = 1;
  await load(0.78, t('load.sun'));
  applyTime(save.set.time);

  // ---------------- game ----------------
  const game = new Game(), input = new Input(), audio = new Audio(), hud = new Hud();
  input.invert = save.set.invert;
  let auto = null, mode = 'menu', paused = false, camMode = Q.get('cam') || 'chase', time = 0, flash = 0;
  const CAMS = ['chase', 'close', 'cockpit', 'cine'];
  const camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), cine = { p: new THREE.Vector3(), t: 0 };
  let camInit = false, fov = 64, lean = 0, lastCtl = { turn: 0, pitch: 0 };
  const touchUI = Q.has('touch') || matchMedia('(pointer: coarse)').matches;
  if (touchUI) input.enableTouch($('touch'));
  $('touch').classList.remove('on');

  function startEvent(id, demo = false) {
    game.load(id, demo ? null : bestOf(id));
    auto = demo ? new Autopilot(game) : null;
    gates.setEvent(game.build); hud.setEvent(game);
    mode = demo ? 'menu' : 'play'; paused = false; camInit = false; audio.streak = 0;
    if (!demo) { camMode = camMode === 'cine' ? 'chase' : camMode; hud.show(true); ['menu', 'pause', 'finish', 'help'].forEach((m) => $(m).classList.remove('on')); if (touchUI) $('touch').classList.add('on'); hintFor(id); }
    else { hud.show(false); game.stateT = 0.01; camMode = 'cine'; cine.t = 0; }
    ghost.root.visible = false;
  }
  function hintFor(id) {
    const first = !save.best[id];
    if (id === 'first' && first) hud.hint(t('hint.first'));
    else if (id === 'free') hud.hint(t('hint.free'));
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
    $('bQual').textContent = t('set.quality') + t('q.' + save.set.quality);
    $('bInv').textContent = t('set.axis') + t(save.set.invert ? 'axis.inv' : 'axis.norm');
    $('bLang').textContent = t('lang.other'); $('bShare').textContent = t('share');
  }
  $('bTime').onclick = () => { const k = Object.keys(TIMES); save.set.time = k[(k.indexOf(save.set.time) + 1) % k.length]; persist(); applyTime(save.set.time); buildCards(); };
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
  $('pRestart').onclick = $('fRetry').onclick = () => startEvent(game.ev.id);
  $('pMenu').onclick = $('fMenu').onclick = toMenu;
  $('fNext').onclick = () => { const i = EVENTS.findIndex((e) => e.id === game.ev.id); startEvent(i >= 0 && i < EVENTS.length - 1 ? EVENTS[i + 1].id : 'free'); };
  function setPause(p) { if (mode !== 'play') return; paused = p; $('pause').classList.toggle('on', p); if (audio.ctx) p ? audio.ctx.suspend() : audio.ctx.resume(); }
  addEventListener('visibilitychange', () => { if (document.hidden && mode === 'play' && game.state === 'fly') setPause(true); });

  function onEvents() {
    for (const e of game.events) {
      if (mode !== 'play') { if (e.type === 'crash') fx.explode(e.p, e.v, e.water); if (e.type === 'respawn' && auto) auto.reset(); if (e.type === 'finish') setTimeout(() => mode === 'menu' && startEvent('tour', true), 2500); continue; }
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
      }
    }
    game.events.length = 0;
  }
  function onFinish(r) {
    const id = game.ev.id;
    if (r.isBest) { save.best[id] = { time: r.time, splits: r.splits, medal: Math.max(r.medal, (save.best[id] || {}).medal || 0), ghost: r.ghost.length < 60000 ? b64(r.ghost) : null, _g: r.ghost }; }
    else save.best[id].medal = Math.max(save.best[id].medal || 0, r.medal);
    try { const c = { ...save, best: Object.fromEntries(Object.entries(save.best).map(([k, v]) => [k, { time: v.time, splits: v.splits, medal: v.medal, ghost: v.ghost }])) }; localStorage.setItem(KEY, JSON.stringify(c)); } catch { /* quota */ }
    $('fname').textContent = evName(game.ev); $('ftime').textContent = fmtTime(r.time);
    $('fmedal').innerHTML = r.medal ? `<i class="medal g${r.medal}" style="width:20px;height:20px"></i> ${t('medal.' + r.medal)}` : t('medal.none');
    $('fbest').innerHTML = r.isBest ? (r.prev ? `${t('fin.newBest')}<span class="num">${fmtTime(r.prev)}</span>` : t('fin.firstBest')) : `${t('fin.best')}<span class="num">${fmtTime(r.prev)}</span>`;
    const m = game.ev.medals, nextT = r.medal < 3 ? m[2 - r.medal] : null;
    $('fstat').innerHTML = `${t('fin.crashes')}${r.crashes} · ${t('fin.misses')}${r.misses}` + (nextT ? ` · ${t('fin.next')}<span class="num">${fmtTime(nextT)}</span>` : '');
    setTimeout(() => { if (mode === 'play' && game.state === 'finished') { $('finish').classList.add('on'); $('touch').classList.remove('on'); } }, 1800);
  }

  // ---------------- cameras ----------------
  const tmpV = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
  camera.worldToScreen = (x, y, z) => { tmpV.set(x, y, z).applyMatrix4(camera.matrixWorldInverse); const behind = tmpV.z > 0; tmpV.applyMatrix4(camera.projectionMatrix); return { x: tmpV.x, y: tmpV.y, behind }; };
  function updateCamera(dt) {
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
      if (cine.t <= 0 || d > 420 || !camInit) {
        // a new vantage point ahead of the bike, beside its path, low over the ground
        const side = Math.random() < 0.5 ? -1 : 1, ahead = 2.2 + Math.random() * 1.5;
        const x = b.p[0] + b.v[0] * ahead + fw[2] * side * (14 + Math.random() * 40) + fh[0] * 40, z = b.p[2] + b.v[2] * ahead - fw[0] * side * (14 + Math.random() * 40) + fh[2] * 40;
        const y = Math.max(surfaceAt(x, z) + 2.5 + Math.random() * 10, b.p[1] + b.v[1] * ahead - 12 + Math.random() * 16);
        cine.p.set(x, y, z); cine.t = 5 + Math.random() * 3; if (!camInit) camLook.copy(tmpV); camInit = true;
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
  const hideInRefl = [model.root, ghost.root, fx.soft.pts, fx.fire.pts, gates.group, gates.beam];
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    // keys
    for (const k of input.pressed) {
      if (k === 'KeyM') { audio.setMuted(!audio.muted); }
      if (mode === 'play') {
        if (k === 'Escape' || k === 'KeyP') { if ($('help').classList.contains('on')) $('help').classList.remove('on'); else setPause(!paused); }
        if (k === 'KeyR') startEvent(game.ev.id);
        if (k === 'KeyC') { camMode = CAMS[(CAMS.indexOf(camMode) + 1) % CAMS.length]; camInit = false; }
        if (k === 'KeyH') $('help').classList.toggle('on');
        if ((k === 'Enter' || k === 'PadA') && game.state === 'finished') startEvent(game.ev.id);
      }
    }
    input.pressed.length = 0;
    if (!paused) {
      time += dt; U.uTime.value = time;
      const ctl = auto ? auto.update() : input.read();
      lastCtl = ctl;
      game.update(dt, ctl);
      onEvents();
      const b = game.bike, alive = game.state !== 'crashed';
      lean += ((ctl.turn || 0) - lean) * Math.min(1, dt * 5);
      model.root.visible = alive; model.update(b, time, lean, dt);
      const gp = mode === 'play' && game.state === 'fly' ? game.ghostAt(game.total) : null;
      ghost.root.visible = !!gp; if (gp) ghost.setPose(gp);
      fx.update(dt, b, game.sim.windW, alive);
      water.wash.set(fx.washPos[0], fx.washPos[1], 6, fx.washStrength);
      gates.update(game, time);
      audio.engine(b, alive && mode === 'play', camera.position.distanceTo(model.root.position), camMode === 'cockpit');
      if (mode === 'play') {
        if (b.abLock && Math.floor(time * 4) % 2 === 0 && ctl.boost) hud.warn(t('warn.heat'));
        else if (b.stall > 0.6 && b.airspeed > 30) hud.warn(t('warn.stall'));
        else if (b.fuel < 8 && Math.floor(time * 2) % 2 === 0) hud.warn(t('warn.fuel'));
      }
      updateCamera(dt);
    }
    // ---- render ----
    const b = game.bike;
    updateTerrain(camera); updateForest(camera, qual.forest);
    sky.position.copy(camera.position); sky.updateMatrix();
    // shadow frustum follows the bike
    const s = sunInfo.sun, fx0 = b.p[0], fy0 = b.p[1], fz0 = b.p[2];
    sun.target.position.set(fx0, fy0, fz0); sun.position.set(fx0 + s[0] * 400, fy0 + s[1] * 400, fz0 + s[2] * 400); sun.target.updateMatrixWorld();
    sun.intensity = s[1] > 0.02 ? Math.PI : 0;
    if (qual.refl) renderReflection(renderer, scene, camera, hideInRefl, (v) => { updateTerrain(v, 2.5); });
    else water.lakes[0].material.uniforms.uHasRefl.value = 0;
    updateTerrain(camera);
    renderer.setRenderTarget(post.main); renderer.clear(); renderer.render(scene, camera);
    if (sun.shadow.map) { U.uShadowMap.value = sun.shadow.map.texture; U.uShadowMat.value.copy(sun.shadow.matrix); U.uShadowOn.value = 1; }
    flash = Math.max(0, flash - dt * 0.5);
    post.finish(camera, s, { blur: camMode === 'cine' ? 0.3 : 0.5, speed: clamp((b.airspeed - 35) / 80, 0, 1) * (camMode === 'cine' ? 0 : 1) + b.abOut * 0.3, time, exposure, flash: Math.min(0.6, flash + Math.max(0, fx.flashT - 0.42) * 6), rays: 1 });
    if (!paused) hud.update(dt, game, camera, viewW, viewH);
    // adaptive resolution
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 1.5) {
      const fps = fpsN / fpsAcc; game.fps = fps; fpsAcc = 0; fpsN = 0;
      if (!Q.has('fixed')) {
        if (fps < 44 && dynScale > 0.6) { dynScale = Math.max(0.6, dynScale - 0.1); resize(); } else if (fps > 58 && dynScale < 1) { dynScale = Math.min(1, dynScale + 0.05); resize(); }
      }
    }
  }

  // ---------------- go ----------------
  await load(0.9, t('load.engines'));
  toMenu();
  if (Q.get('event')) { startEvent(Q.get('event')); if (Q.has('demo')) { auto = new Autopilot(game); game.stateT = 0.01; } }
  frame(performance.now());
  await load(1, '');
  $('load').classList.add('off');
  addEventListener('pointerdown', () => audio.start(), { once: true }); addEventListener('keydown', () => audio.start(), { once: true });
  window.game = game; window.dbg = { scene, camera, renderer, post, setCam: (m) => { camMode = m; camInit = false; }, free: (x, y, z, lx, ly, lz, f = 55) => { camMode = 'free'; camera.position.set(x, y, z); camera.up.set(0, 1, 0); camera.lookAt(lx, ly, lz); fov = f; }, hud: (on) => hud.show(on), applyTime, get mode() { return mode; }, startEvent, U, sun };
}
if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
boot().catch((e) => { const el = $('err'); el.style.display = 'block'; el.textContent += (e.stack || e) + '\n'; console.error(e); });
