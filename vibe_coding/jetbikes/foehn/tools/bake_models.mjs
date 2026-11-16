// Turns the downloaded models (CC0, see assets/models/CREDITS.md) into one small file the game
// loads: assets/models/models.bin. Every model becomes plain triangles with a colour per corner,
// its feet on y = 0, its nose towards +x, in metres. An animated animal is frozen in a handful
// of poses (standing, grazing, four moments of the gallop), so the game can draw hundreds of
// them as instances without a skeleton.
//
//   THREE_DIR=/path/to/node_modules/three node tools/bake_models.mjs [cache-dir]
//
// The .glb files are fetched into the cache directory once. three.js with its examples is needed
// only here (the game itself ships the bare library).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CACHE = process.argv[2] || path.join(ROOT, '.model-cache');
const TD = process.env.THREE_DIR; if (!TD) { console.error('set THREE_DIR to a three.js package (npm i three@0.169)'); process.exit(1); }
globalThis.self = globalThis;
const { PNG } = await import(pathToFileURL(path.join(TD, '../pngjs/lib/png.js'))).then((m) => m.default || m);
const JPEG = await import(pathToFileURL(path.join(TD, '../jpeg-js/index.js'))).then((m) => m.default || m);
const THREE = await import(pathToFileURL(path.join(TD, 'build/three.module.js')));
const { GLTFLoader } = await import(pathToFileURL(path.join(TD, 'examples/jsm/loaders/GLTFLoader.js')));

const U = 'https://static.poly.pizza/';
// poses: name -> [clip, moment 0..1 of the clip]. h: height in metres of the standing animal.
const GALLOP = (clip) => ({ run0: [clip, 0], run1: [clip, 0.25], run2: [clip, 0.5], run3: [clip, 0.75] });
const HERD = (h) => ({ h, poses: { stand: ['Idle', 0.3], graze: ['Eating', 0.5], ...GALLOP('Gallop') } });
export const MODELS = {
  cow: { id: '26zM1outCr', ...HERD(1.55) },
  bull: { id: 'a8PIIYwF7r', ...HERD(1.7) },
  holstein: { id: '5XSc2Fka3F', h: 1.6, poses: { stand: ['Idle', 0.2], graze: ['Idle', 0.6], ...GALLOP('Run') } },
  sheep: { id: 'C39AUXUUes', h: 1.0, poses: { stand: ['Idle', 0.2], graze: ['Idle', 0.6], ...GALLOP('Jump') } },
  deer: { id: 'T6Cs7tmMHJ', ...HERD(1.45) },
  stag: { id: 'tQdzbZ1Cmw', ...HERD(2.0) },
  horse: { id: 'qvTrSG9pZF', ...HERD(1.95) },
  whitehorse: { id: 'bEdE4rmZy9', ...HERD(1.95) },
};
// things that stand still: one shape each. h: height in metres (the game scales every one it places).
Object.assign(MODELS, {
  rockA: { id: 'KZdEP3uUpa', h: 1 }, rockB: { id: 'JQxF95498B', h: 1 }, rockC: { id: 's1OJ3bBzqc', h: 1 }, shards: { id: 'JmFMh7ztL9', h: 1 },
  bush: { id: '92EytlU1El', h: 1 }, berries: { id: 'TSbIxkDtxF', h: 1 }, flowers: { id: 'dOO6kMDd8L', h: 1 },
  stump: { id: 'nFvEbUX6LE', h: 1 }, log: { id: 'nwsYvcI0bC', h: 1 }, hay: { id: 'Yu8TOERkpw', h: 1 },
});
// buildings. lit: the materials that are window panes (they glow at night). Their proportions go
// into js/sim/buildings.js, so that what the bike can hit is what the rider sees.
const BUILDINGS = {
  houseA: { id: 'RSwoYSLblu', h: 1 }, houseB: { id: 'oJJIRwv6Bo', h: 1 }, houseC: { id: 'roqiHdrpgc', h: 1, lit: ['DarkGrey'] }, houseD: { id: 'HeHDd2rTpX', h: 1, lit: ['DarkGrey'] }, houseE: { id: 'L7h0SjZX2K', h: 1 },
  hut: { id: 'wxi3kAu5ey', h: 1 }, barnA: { id: 'vSqQNA7ez6', h: 1 }, barnB: { id: 'aTcDP7UERC', h: 1 }, well: { id: 'QlqncKYxXb', h: 1 },
};
Object.assign(MODELS, BUILDINGS);
const URLS = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets/models/sources.json'), 'utf8'));

async function load(id) {
  fs.mkdirSync(CACHE, { recursive: true });
  const f = path.join(CACHE, id + '.glb');
  if (!fs.existsSync(f)) { const r = await fetch(URLS[id].url, { headers: { 'User-Agent': 'foehn-bake' } }); if (!r.ok) throw new Error(`${id}: ${r.status}`); fs.writeFileSync(f, Buffer.from(await r.arrayBuffer())); }
  const buf = fs.readFileSync(f);
  const gltf = await new Promise((res, rej) => new GLTFLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '', res, rej));
  // the pictures the materials are painted with (the loader cannot decode them outside a browser):
  // the game keeps one colour per corner, so each corner takes the colour the picture has there
  gltf.paint = {};
  const J = gltf.parser.json, n0 = buf.readUInt32LE(12), bin = buf.subarray(20 + n0 + 8);
  for (const m of J.materials || []) {
    const t = m.pbrMetallicRoughness && m.pbrMetallicRoughness.baseColorTexture; if (!t) continue;
    const im = J.images[J.textures[t.index].source]; if (im.bufferView === undefined) continue;
    const bv = J.bufferViews[im.bufferView], bytes = bin.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
    try { const d = im.mimeType === 'image/jpeg' ? JPEG.decode(bytes, { useTArray: true }) : PNG.sync.read(Buffer.from(bytes)); gltf.paint[m.name] = { w: d.width, h: d.height, d: d.data, blend: m.alphaMode === 'BLEND' || m.alphaMode === 'MASK' }; } catch (e) { console.warn(id, m.name, 'picture not read:', e.message); }
  }
  return gltf;
}
// the picture's colour around (u, v), linear: a small patch is averaged so that one odd texel does not paint a whole face
function sample(t, u, v) {
  u -= Math.floor(u); v -= Math.floor(v);
  const cx = Math.min(t.w - 1, Math.floor(u * t.w)), cy = Math.min(t.h - 1, Math.floor(v * t.h)), R = Math.max(1, Math.round(t.w / 128));
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = cy - R; y <= cy + R; y++) for (let x = cx - R; x <= cx + R; x++) { const k = (((y + t.h) % t.h) * t.w + ((x + t.w) % t.w)) * 4; if (t.d[k + 3] < 128) continue; r += lin(t.d[k] / 255); g += lin(t.d[k + 1] / 255); b += lin(t.d[k + 2] / 255); n++; }
  return n ? [r / n, g / n, b / n] : [0.2, 0.3, 0.1];
}
const lin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

// the whole scene as it stands now: positions (world), colours, indices
function freeze(scene, paint = {}) {
  scene.updateMatrixWorld(true);
  const pos = [], col = [], idx = [], lit = [], v = new THREE.Vector3();
  scene.traverse((o) => {
    if (!o.isMesh) return;
    if (o.isSkinnedMesh) o.skeleton.update();
    const g = o.geometry, P = g.attributes.position, C = g.attributes.color, UV = g.attributes.uv, base = pos.length / 3;
    const mats = [].concat(o.material), groups = g.groups.length ? g.groups : [{ start: 0, count: Infinity, materialIndex: 0 }];
    const matOf = new Int32Array(P.count).fill(0), I = g.index;
    for (const gr of groups) { const n = Math.min(gr.count, (I ? I.count : P.count) - gr.start); for (let k = gr.start; k < gr.start + n; k++) matOf[I ? I.getX(k) : k] = gr.materialIndex; }
    for (let i = 0; i < P.count; i++) {
      if (o.isSkinnedMesh) o.getVertexPosition(i, v); else v.fromBufferAttribute(P, i);
      v.applyMatrix4(o.matrixWorld); pos.push(v.x, v.y, v.z);
      const m = mats[matOf[i]] || mats[0], c = m.color;                     // (three keeps colours linear)
      let r = c.r, gg = c.g, b = c.b;
      if (C && m.vertexColors) { r *= C.getX(i); gg *= C.getY(i); b *= C.getZ(i); }
      const t = paint[m.name]; if (t && UV) { const q = sample(t, UV.getX(i), UV.getY(i)); r *= q[0]; gg *= q[1]; b *= q[2]; }
      col.push(r, gg, b); lit.push(m.name);
    }
    const n = I ? I.count : P.count;
    for (let k = 0; k < n; k++) idx.push(base + (I ? I.getX(k) : k));
  });
  return { pos, col, idx, lit };
}

const out = { models: {} }, chunks = []; let at = 0;
const put = (arr) => { const b = Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength), pad = (4 - (b.length & 3)) & 3; chunks.push(b, Buffer.alloc(pad)); const o = at; at += b.length + pad; return o; };
for (const [name, M] of Object.entries(MODELS)) {
  const gltf = await load(M.id), mixer = new THREE.AnimationMixer(gltf.scene);
  const clip = (n) => gltf.animations.find((a) => a.name === n) || gltf.animations.find((a) => a.name.endsWith('|' + n)) || (() => { throw new Error(`${name}: no clip ${n} in ${gltf.animations.map((a) => a.name)}`); })();
  let fit = null; const rec = { poses: {} };
  for (const [pose, def] of Object.entries(M.poses || { still: null })) {
    if (def) { mixer.stopAllAction(); const c = clip(def[0]), a = mixer.clipAction(c); a.play(); mixer.setTime(def[1] * c.duration); }
    const f = freeze(gltf.scene, gltf.paint), n = f.pos.length / 3;
    if (!fit) {                                           // the first pose (standing) sets the size and the place for all of them
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, z0 = 1e9, z1 = -1e9;
      for (let i = 0; i < n; i++) { x0 = Math.min(x0, f.pos[i * 3]); x1 = Math.max(x1, f.pos[i * 3]); y0 = Math.min(y0, f.pos[i * 3 + 1]); y1 = Math.max(y1, f.pos[i * 3 + 1]); z0 = Math.min(z0, f.pos[i * 3 + 2]); z1 = Math.max(z1, f.pos[i * 3 + 2]); }
      fit = { s: M.h / (y1 - y0), cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, y0 };
      rec.len = (z1 - z0) * fit.s; rec.h = M.h; rec.wid = (x1 - x0) * fit.s;
    }
    // glTF looks down +z: turn it to the game's +x (x' = z, z' = -x)
    const P = new Int16Array(n * 3), C = new Uint8Array(n * 4), Q = 2000;      // positions in half millimetres
    for (let i = 0; i < n; i++) {
      const x = (f.pos[i * 3] - fit.cx) * fit.s, y = (f.pos[i * 3 + 1] - fit.y0) * fit.s, z = (f.pos[i * 3 + 2] - fit.cz) * fit.s;
      P[i * 3] = Math.round(z * Q); P[i * 3 + 1] = Math.round(y * Q); P[i * 3 + 2] = Math.round(-x * Q);
      for (let k = 0; k < 3; k++) C[i * 4 + k] = Math.round(255 * Math.sqrt(Math.min(1, Math.max(0, f.col[i * 3 + k]))));   // stored as the square root, squared again in the shader
      C[i * 4 + 3] = M.lit && M.lit.includes(f.lit[i]) ? 0 : 255;                                                            // 0: a window pane
    }
    // the poses of one animal are the same triangles moved: colours and indices are kept once
    if (rec.n === undefined) { const I = n > 65535 ? new Uint32Array(f.idx) : new Uint16Array(f.idx); Object.assign(rec, { n, q: Q, c: put(C), i: put(I), ni: I.length, wide: n > 65535 }); }
    else if (rec.n !== n) throw new Error(`${name}: pose ${pose} has another number of corners`);
    rec.poses[pose] = put(P);
  }
  out.models[name] = rec;
  console.log(name.padEnd(11), Object.keys(rec.poses).length, 'poses,', rec.ni / 3, 'triangles, length', rec.len.toFixed(2), 'm');
}
fs.writeFileSync(path.join(ROOT, 'js/sim/buildings.js'), `// Written by tools/bake_models.mjs: the proportions of the building models, for a height of 1.\n// [length along the building's x, height, width along its z]\nexport const SHAPES = ${JSON.stringify(Object.fromEntries(Object.keys(BUILDINGS).map((k) => [k, [+out.models[k].len.toFixed(3), 1, +out.models[k].wid.toFixed(3)]])), null, 1)};\n`);
const head = Buffer.from(JSON.stringify(out)), hp = Buffer.alloc((4 - (head.length & 3)) & 3, 32);
const len = Buffer.alloc(4); len.writeUInt32LE(head.length + hp.length);
fs.writeFileSync(path.join(ROOT, 'assets/models/models.bin'), Buffer.concat([len, head, hp, ...chunks]));
console.log('models.bin', ((4 + head.length + hp.length + at) / 1024).toFixed(0), 'KB');
