/* חזיתות - faces for the houses of the 3D background (model3d.js).
 *
 * Ori, 14/10/2026: "right now everything looks like plain boxes; give the
 * houses textures, taken from Google Street View."
 *
 * Not from Street View: the Google Maps Platform terms forbid making
 * anything derived from its imagery - textures and models by name - and
 * forbid keeping the pictures at all. pardes3d ran into the same wall and
 * settled it the same way (see its project-knowledge, rule 2). Real
 * photographs can come later from Mapillary, which is CC BY-SA.
 *
 * So the faces are drawn, not photographed: a fragment shader that knows
 * what a house in the moshava is made of. Plaster with its stains and the
 * dirt the rain splashes up, a plinth, a window per three metres on every
 * floor with its sill, its frame and its aluminium roller shutter pulled
 * down a different way in each window; on the old low houses wooden
 * shutters painted green or blue standing open; on the blocks of flats a
 * balcony with a railing; a door on the long side; a coping along the top.
 * Red-roofed houses that are four-sided get a hipped tile roof, and the flat
 * roofs get what every flat roof in Israel has - a solar water heater, a
 * tank and two panels facing south.
 *
 * The buildings are the same footprints and heights MapLibre extruded; they
 * are drawn by Three.js as a custom layer in the map, sharing its depth
 * buffer, so the trees and walls that MapLibre still draws stand in front
 * of and behind them correctly. While Three.js is on its way, or if it never
 * comes, the plain extrusions stay.
 *
 * Coordinates: the vertices are in Web Mercator relative to the model's
 * origin, divided by the length of a metre there, so that one unit is about
 * a metre and the float32s on the GPU have plenty of precision even from
 * a pavement. Mercator and not plain metres around the origin because over
 * the six kilometres of the moshava the two drift apart by tens of metres,
 * and the trees would stand in the walls. Heights are scaled by the
 * mercator metre at each house's own latitude, which is what MapLibre does.
 * The facade's own coordinates, the ones the windows are laid out in, are
 * true metres.
 */
'use strict';

const Facades = (() => {
  const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
  const RAD = Math.PI / 180;
  const M_LAT = 111320;
  const RED_ROOF = 3;                 // index of the tile colour in roof_colors
  // The style's light: azimuth 215, 40 degrees from the zenith (app.js).
  const SUN = (() => {
    const az = 215 * RAD, el = 50 * RAD;
    return [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)];
  })();

  let THREE = null;
  let built = null;       // {scene, origin, s0} - made once, kept across style swaps
  let renderer = null;
  let building = null;

  /* ---------- geometry ---------- */

  const merc = (lng, lat) => [(lng + 180) / 360,
    (180 - (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + lat * RAD / 2))) / 360];
  const metreAt = (lat) => 1 / (40075016.686 * Math.cos(lat * RAD));

  function rnd(seed) {
    let s = seed >>> 0 || 1;
    return () => {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  /** A closed counter-clockwise ring moved `eps` outward (inward if
   *  negative) along the mitre at each corner, so the corners stay closed. */
  function grow(P, eps) {
    const n = P.length, Q = [];
    for (let i = 0; i < n; i++) {
      const a = P[(i + n - 1) % n], b = P[i], c = P[(i + 1) % n];
      const ux = b[0] - a[0], uy = b[1] - a[1], vx = c[0] - b[0], vy = c[1] - b[1];
      const ul = Math.hypot(ux, uy) || 1, vl = Math.hypot(vx, vy) || 1;
      const n1x = uy / ul, n1y = -ux / ul, n2x = vy / vl, n2y = -vx / vl;
      const k = eps / Math.max(1 + n1x * n2x + n1y * n2y, 0.3);
      Q.push([b[0] + (n1x + n2x) * k, b[1] + (n1y + n2y) * k]);
    }
    return Q;
  }

  /** Grows typed arrays as vertices are pushed. */
  function buffers(spec) {
    const out = {};
    for (const [k, n] of Object.entries(spec)) out[k] = { n, a: new Float32Array(n * 4096), len: 0 };
    out.push = (k, ...v) => {
      const b = out[k];
      if (b.len + v.length > b.a.length) {
        const g = new Float32Array(b.a.length * 2);
        g.set(b.a);
        b.a = g;
      }
      for (const x of v) b.a[b.len++] = x;
    };
    return out;
  }

  /* Some 1.9 million vertices: on a phone the bytes matter. Colours go to
   * the GPU as bytes and normals as signed bytes, and once a buffer is on
   * the GPU the copy in JavaScript is let go - nothing reads it again (no
   * raycasting here: model3d.js has its own index of the walls). */
  function bytes(a, Out, k) {
    const out = new Out(a.length);
    for (let i = 0; i < a.length; i++) out[i] = Math.round(a[i] * k);
    return out;
  }
  const PACK = {
    col: (a) => bytes(a, Uint8ClampedArray, 255),
    nrm: (a) => bytes(a, Int8Array, 127),
    nrm3: (a) => bytes(a, Int8Array, 127)
  };
  function release() { this.array = null; }

  function geometryFrom(buf, keys, index) {
    const g = new THREE.BufferGeometry();
    for (const k of keys) {
      const b = buf[k];
      const a = b.a.subarray(0, b.len);
      const attr = PACK[k] ? new THREE.BufferAttribute(PACK[k](a), b.n, true)
        : new THREE.BufferAttribute(a.slice(), b.n);
      g.setAttribute(k, attr.onUpload(release));
      b.a = null;
    }
    if (index) {
      const ix = new THREE.BufferAttribute(Uint32Array.from(index), 1);
      g.setIndex(ix.onUpload(release));
    }
    g.computeBoundingSphere();
    return g;
  }

  /* The build takes one to two seconds on a laptop and several on a phone,
   * so it gives the page back between chunks rather than freezing the map
   * while the faces are made. */
  let lastYield = 0;
  async function breathe() {
    if (performance.now() - lastYield < 40) return;
    await new Promise((r) => setTimeout(r, 0));
    lastYield = performance.now();
  }

  async function build(raw, rawWalls) {
    const [ox, oy] = merc(raw.origin[0], raw.origin[1]);
    const s0 = metreAt(raw.origin[1]);
    const toLocal = (lng, lat) => {
      const [mx, my] = merc(lng, lat);
      return [(mx - ox) / s0, -(my - oy) / s0];
    };

    // walls: position, facade (u, v, edge length, height of the floors),
    // normal, colour, aux (seed, edge index + 1000 on the front, kind,
    // parapet height or -1 under a hipped roof). Kinds: 0 a shed or the
    // inside of a parapet, 1 a house, 2 a block of flats; 3, 4 and 5 the
    // garden walls - plastered, concrete block, hedge.
    const W = buffers({ position: 3, facade: 4, nrm: 3, col: 3, aux: 4 });
    const wIdx = [];
    // roofs: position, normal, colour, aux (seed, tile?, slope-u, slope-v)
    const R = buffers({ position: 3, nrm3: 3, col: 3, aux: 4 });
    const rIdx = [];
    // solar water heaters: position, normal, colour
    const H = buffers({ position: 3, nrm3: 3, col: 3 });
    const hIdx = [];

    const wallCol = raw.wall_colors.map((c) => c.map((x) => x / 255));
    const roofCol = raw.roof_colors.map((c) => c.map((x) => x / 255));

    let nb = 0;
    lastYield = performance.now();
    for (const row of raw.rows) {
      await breathe();
      const h = row[0], wc = wallCol[row[1]], rc = roofCol[row[2]];
      // unpack the footprint
      const ll = [];
      let x = 0, y = 0;
      for (let i = 3; i + 1 < row.length; i += 2) {
        x += row[i]; y += row[i + 1];
        ll.push([raw.origin[0] + x / raw.q, raw.origin[1] + y / raw.q]);
      }
      if (ll.length < 3) continue;
      const lat = ll[0][1];
      const zk = metreAt(lat) / s0;          // a metre of height in local units
      const kx = M_LAT * Math.cos(lat * RAD);
      let P = ll.map(([a, b]) => toLocal(a, b));
      let M = ll.map(([a, b]) => [a * kx, b * M_LAT]);   // true metres, for lengths
      // drop a closing duplicate and zero-length edges
      const keep = [];
      for (let i = 0; i < P.length; i++) {
        const j = (i + 1) % P.length;
        if (Math.hypot(M[j][0] - M[i][0], M[j][1] - M[i][1]) > 0.05) keep.push(i);
      }
      P = keep.map((i) => P[i]); M = keep.map((i) => M[i]);
      if (P.length < 3) continue;
      // counter-clockwise, so that the outward normal of edge i->j is (dy, -dx)
      let area = 0;
      for (let i = 0; i < P.length; i++) {
        const j = (i + 1) % P.length;
        area += P[i][0] * P[j][1] - P[j][0] * P[i][1];
      }
      if (area < 0) { P.reverse(); M.reverse(); area = -area; }
      area /= 2;
      const seed = nb++;
      const r = rnd(seed * 2654435761 + 7);
      const sd = r();

      // OSM and Microsoft's footprints overlap here and there - some 470
      // pairs, ten of them nearly the same house twice - and where two
      // share the line of a wall, two facades with different windows lie
      // in one plane and the depth buffer would flicker between them. So
      // every house stands a few centimetres proud of its footprint, by its
      // own amount, and in any such pair one of them is in front.
      P = grow(P, (0.03 + 0.09 * sd) * zk);

      // A hipped tile roof for a red, four-sided house that is not too big.
      let eaves = h, hip = null;
      if (row[2] === RED_ROOF && P.length === 4 && area < 450 && h < 11) {
        const L = (i) => Math.hypot(M[(i + 1) % 4][0] - M[i][0], M[(i + 1) % 4][1] - M[i][1]);
        const long = L(0) + L(2) >= L(1) + L(3) ? 0 : 1;
        const wide = (L(long + 1) + L((long + 3) % 4)) / 2;
        const rise = Math.min(wide / 2 * Math.tan(24 * RAD), 3);
        eaves = Math.max(2.8, h - rise * 0.6);
        hip = { long, wide, rise };
      }

      // The front: the longest edge gets the door.
      let front = 0, best = 0;
      const edgeLen = [];
      for (let i = 0; i < P.length; i++) {
        const j = (i + 1) % P.length;
        const L = Math.hypot(M[j][0] - M[i][0], M[j][1] - M[i][1]);
        edgeLen.push(L);
        if (L > best) { best = L; front = i; }
      }
      const kind = h < 2.6 ? 0 : (h >= 8.5 ? 2 : 1);   // shed, house, block of flats
      // A flat roof has a parapet round it, half a metre to a metre of wall
      // above the roof with a coping on top: from the street it is what
      // makes the top of a house a top and not the edge of a box.
      const par = !hip && kind > 0 ? 0.45 + 0.55 * r() : 0;
      // overlapping flat roofs get the same few centimetres as the walls
      const rz = (eaves + 0.06 * sd) * zk;
      const wallQuad = (a, b, u0, L, v0, v1, z0, z1, nx, ny, edge, k, col, w4) => {
        const base = W.position.len / 3;
        for (const [px, py, u] of [[a[0], a[1], u0], [b[0], b[1], u0 + L]]) {
          for (const [v, z] of [[v0, z0], [v1, z1]]) {
            W.push('position', px, py, z);
            W.push('facade', u, v, L, eaves);
            W.push('nrm', nx, ny, 0);
            W.push('col', col[0], col[1], col[2]);
            W.push('aux', sd, edge, k, w4);
          }
        }
        wIdx.push(base, base + 2, base + 3, base, base + 3, base + 1);
      };
      const inner = par ? grow(P, -0.22 * zk) : null;
      const capCol = [wc[0] * 1.03, wc[1] * 1.03, wc[2] * 1.03];
      for (let i = 0; i < P.length; i++) {
        const j = (i + 1) % P.length;
        const L = edgeLen[i];
        const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], dl = Math.hypot(dx, dy) || 1;
        const nx = dy / dl, ny = -dx / dl;
        const edge = i + (i === front ? 1000 : 0);
        wallQuad(P[i], P[j], 0, L, 0, eaves + par, 0, rz + par * zk, nx, ny, edge, kind, wc,
          hip ? -1 : par);
        if (par) {
          // the parapet's inside face, and the coping that joins the two
          wallQuad(inner[j], inner[i], 0, L, eaves, eaves + par, rz, rz + par * zk, -nx, -ny,
            edge, 0, wc, par);
          const base = R.position.len / 3;
          for (const p of [P[i], P[j], inner[j], inner[i]]) {
            R.push('position', p[0], p[1], rz + par * zk);
            R.push('nrm3', 0, 0, 1);
            R.push('col', ...capCol);
            R.push('aux', sd, 0, p[0], p[1]);
          }
          rIdx.push(base, base + 1, base + 2, base, base + 2, base + 3);
        }
      }

      // roof
      const roofV = (px, py, pz, n, tile, su, sv) => {
        R.push('position', px, py, pz);
        R.push('nrm3', n[0], n[1], n[2]);
        R.push('col', rc[0], rc[1], rc[2]);
        R.push('aux', sd, tile, su, sv);
      };
      if (hip) {
        const { long, wide, rise } = hip;
        const a = P[long], b = P[(long + 1) % 4], c = P[(long + 2) % 4], d = P[(long + 3) % 4];
        const ma = M[long], mb = M[(long + 1) % 4];
        const along = Math.hypot(mb[0] - ma[0], mb[1] - ma[1]);
        const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
        const e0 = mid(a, d), e1 = mid(b, c);
        const ex = e1[0] - e0[0], ey = e1[1] - e0[1], el = Math.hypot(ex, ey) || 1;
        const inset = Math.min(wide / 2, along * 0.45) * (el / along);
        const r0 = [e0[0] + ex / el * inset, e0[1] + ey / el * inset];
        const r1 = [e1[0] - ex / el * inset, e1[1] - ey / el * inset];
        const top = rz + rise * zk;
        const face = (pts) => {
          // normal of the plane through the first three points
          const [p, q, s] = pts;
          const ux = q[0] - p[0], uy = q[1] - p[1], uz = q[2] - p[2];
          const vx = s[0] - p[0], vy = s[1] - p[1], vz = s[2] - p[2];
          let n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
          const nl = Math.hypot(...n) || 1;
          n = n.map((t) => t / nl);
          if (n[2] < 0) n = n.map((t) => -t);
          // tile rows run along the eave: u along it, v up the slope, in metres
          const ex2 = q[0] - p[0], ey2 = q[1] - p[1], eL = Math.hypot(ex2, ey2) || 1;
          const base = R.position.len / 3;
          for (const pt of pts) {
            const rx = pt[0] - p[0], ry = pt[1] - p[1];
            const su = (rx * ex2 + ry * ey2) / eL;
            const horiz = Math.abs(rx * ey2 - ry * ex2) / eL;
            const sv = Math.hypot(horiz, (pt[2] - p[2]) / zk);
            roofV(pt[0], pt[1], pt[2], n, 1, su, sv);
          }
          for (let k = 1; k + 1 < pts.length; k++) rIdx.push(base, base + k, base + k + 1);
        };
        const A = [a[0], a[1], rz], B = [b[0], b[1], rz], C = [c[0], c[1], rz], D = [d[0], d[1], rz];
        const R0 = [r0[0], r0[1], top], R1 = [r1[0], r1[1], top];
        face([A, B, R1, R0]);
        face([C, D, R0, R1]);
        face([B, C, R1]);
        face([D, A, R0]);
      } else {
        const contour = P.map(([px, py]) => new THREE.Vector2(px, py));
        let tris;
        try {
          tris = THREE.ShapeUtils.triangulateShape(contour, []);
        } catch (err) {
          tris = [];
        }
        const base = R.position.len / 3;
        const tile = row[2] === RED_ROOF ? 1 : 0;
        for (const [px, py] of P) roofV(px, py, rz, [0, 0, 1], tile, px, py);
        for (const t of tris) rIdx.push(base + t[0], base + t[1], base + t[2]);

        // A solar water heater on most flat roofs of houses and small blocks.
        if (!tile && h >= 2.6 && h < 16 && area > 45 && area < 900 && r() < 0.75) {
          let cx = 0, cy = 0;
          for (const [px, py] of P) { cx += px; cy += py; }
          cx /= P.length; cy /= P.length;
          // nudge towards a random corner, but keep it on the roof
          const k = P[Math.floor(r() * P.length)];
          const t = 0.25 + r() * 0.3;
          const hx = cx + (k[0] - cx) * t, hy = cy + (k[1] - cy) * t;
          heater(H, hIdx, hx, hy, rz, zk, r);
          if (area > 300 && r() < 0.6) {
            const k2 = P[Math.floor(r() * P.length)];
            heater(H, hIdx, cx + (k2[0] - cx) * 0.35, cy + (k2[1] - cy) * 0.35, rz, zk, r);
          }
        }
      }
    }

    if (rawWalls) await gardenWalls(rawWalls, toLocal, s0, W, wIdx);

    const scene = new THREE.Scene();
    const uni = {
      uSun: { value: new THREE.Vector3(...SUN) },
      uCam: { value: new THREE.Vector3() },
      uFog: { value: new THREE.Color('#dfe6ea') },
      uSky: { value: new THREE.Color('#9ec3e6') }
    };
    const walls = new THREE.Mesh(geometryFrom(W, ['position', 'facade', 'nrm', 'col', 'aux'], wIdx),
      new THREE.ShaderMaterial({ uniforms: uni, vertexShader: WALL_VS, fragmentShader: WALL_FS,
        side: THREE.DoubleSide, extensions: { derivatives: true } }));
    const roofs = new THREE.Mesh(geometryFrom(R, ['position', 'nrm3', 'col', 'aux'], rIdx),
      new THREE.ShaderMaterial({ uniforms: uni, vertexShader: ROOF_VS, fragmentShader: ROOF_FS,
        side: THREE.DoubleSide, extensions: { derivatives: true } }));
    const heaters = new THREE.Mesh(geometryFrom(H, ['position', 'nrm3', 'col'], hIdx),
      new THREE.ShaderMaterial({ uniforms: uni, vertexShader: LIT_VS, fragmentShader: LIT_FS }));
    for (const m of [walls, roofs, heaters]) {
      m.frustumCulled = false;
      m.matrixAutoUpdate = false;
      scene.add(m);
    }
    return { scene, uni, ox, oy, s0, count: nb,
      tris: (wIdx.length + rIdx.length + hIdx.length) / 3 };
  }

  /** The walls round the gardens, 20 thousand of them along the parcel
   *  lines: each a polyline with a height and a thickness, built as two
   *  faces and a top, with ends. They go into the walls' buffers with kinds
   *  3 to 5, so the same shader dresses them - in plaster with a coping,
   *  in bare concrete block, or in leaves. */
  async function gardenWalls(raw, toLocal, s0, W, idx) {
    const COL = [[0.89, 0.86, 0.8], [0.66, 0.65, 0.61], [0.25, 0.4, 0.18]];
    let seed = 0;
    for (const row of raw.rows) {
      await breathe();
      const [kind, h, thick] = row;
      const pts = [];
      let x = 0, y = 0;
      for (let i = 3; i + 1 < row.length; i += 2) {
        x += row[i]; y += row[i + 1];
        const lng = raw.origin[0] + x / raw.q, lat = raw.origin[1] + y / raw.q;
        const p = toLocal(lng, lat);
        const last = pts[pts.length - 1];
        if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.05) pts.push(p);
      }
      if (pts.length < 2) continue;
      const lat = raw.origin[1] + y / raw.q;
      const zk = metreAt(lat) / s0;
      const sd = ((seed++ * 2654435761) >>> 0) / 4294967296;
      const k = 3 + Math.min(kind, 2), col = COL[Math.min(kind, 2)];
      const half = thick / 2 * zk, top = h * zk;
      // the two sides, offset along the averaged normal at each vertex
      const Lf = [], Rt = [], along = [0];
      for (let i = 0; i < pts.length; i++) {
        const a = pts[Math.max(i - 1, 0)], b = pts[Math.min(i + 1, pts.length - 1)];
        let dx = b[0] - a[0], dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        dx /= len; dy /= len;
        Lf.push([pts[i][0] - dy * half, pts[i][1] + dx * half]);
        Rt.push([pts[i][0] + dy * half, pts[i][1] - dx * half]);
        if (i) along.push(along[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) / zk);
      }
      const total = along[along.length - 1];
      const quad = (p, uv, n) => {
        const base = W.position.len / 3;
        for (let q = 0; q < 4; q++) {
          W.push('position', p[q][0], p[q][1], p[q][2]);
          W.push('facade', uv[q][0], uv[q][1], total, h);
          W.push('nrm', n[0], n[1], n[2]);
          W.push('col', col[0], col[1], col[2]);
          W.push('aux', sd, 0, k, 0);
        }
        idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
      };
      for (let i = 0; i + 1 < pts.length; i++) {
        const dx = pts[i + 1][0] - pts[i][0], dy = pts[i + 1][1] - pts[i][1];
        const dl = Math.hypot(dx, dy) || 1;
        const nx = dy / dl, ny = -dx / dl;     // to the right of the line
        const u0 = along[i], u1 = along[i + 1];
        const l0 = Lf[i], l1 = Lf[i + 1], r0 = Rt[i], r1 = Rt[i + 1];
        quad([[r0[0], r0[1], 0], [r1[0], r1[1], 0], [r1[0], r1[1], top], [r0[0], r0[1], top]],
          [[u0, 0], [u1, 0], [u1, h], [u0, h]], [nx, ny, 0]);
        quad([[l1[0], l1[1], 0], [l0[0], l0[1], 0], [l0[0], l0[1], top], [l1[0], l1[1], top]],
          [[u1, 0], [u0, 0], [u0, h], [u1, h]], [-nx, -ny, 0]);
        quad([[r0[0], r0[1], top], [r1[0], r1[1], top], [l1[0], l1[1], top], [l0[0], l0[1], top]],
          [[u0, h], [u1, h], [u1, h + thick], [u0, h + thick]], [0, 0, 1]);
      }
      // the two ends
      for (const [i, j] of [[0, 1], [pts.length - 1, pts.length - 2]]) {
        const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];   // outward
        const dl = Math.hypot(dx, dy) || 1;
        const a = Lf[i], b = Rt[i], u = along[i];
        quad([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], top], [a[0], a[1], top]],
          [[u, 0], [u + thick, 0], [u + thick, h], [u, h]], [dx / dl, dy / dl, 0]);
      }
    }
  }

  /** A solar water heater: a white tank lying on a frame, and two dark
   *  collector panels in front of it tilted 40 degrees to the south. */
  function heater(H, idx, x, y, z0, zk, r) {
    const yaw = (r() - 0.5) * 0.5;     // roughly south, never quite
    const cs = Math.cos(yaw), sn = Math.sin(yaw);
    // local frame: u east-west, v north(+)/south(-), w up; metres
    const put = (u, v, w) => [x + u * cs - v * sn, y + u * sn + v * cs, z0 + w * zk];
    const rot = (n) => [n[0] * cs - n[1] * sn, n[0] * sn + n[1] * cs, n[2]];
    const quad = (p, n, c) => {
      const base = H.position.len / 3;
      const nn = rot(n);
      for (const q of p) {
        H.push('position', ...put(...q));
        H.push('nrm3', ...nn);
        H.push('col', ...c);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    };
    const box = (u0, u1, v0, v1, w0, w1, c) => {
      quad([[u0, v0, w1], [u1, v0, w1], [u1, v1, w1], [u0, v1, w1]], [0, 0, 1], c);
      quad([[u0, v0, w0], [u1, v0, w0], [u1, v0, w1], [u0, v0, w1]], [0, -1, 0], c);
      quad([[u1, v1, w0], [u0, v1, w0], [u0, v1, w1], [u1, v1, w1]], [0, 1, 0], c);
      quad([[u0, v1, w0], [u0, v0, w0], [u0, v0, w1], [u0, v1, w1]], [-1, 0, 0], c);
      quad([[u1, v0, w0], [u1, v1, w0], [u1, v1, w1], [u1, v0, w1]], [1, 0, 0], c);
    };
    const n = r() < 0.7 ? 2 : 1;
    const pw = 1.0, pl = 2.0, tilt = 40 * RAD;
    const ph = Math.sin(tilt) * pl, pd = Math.cos(tilt) * pl;
    const total = n * pw + (n - 1) * 0.08;
    // panels: from the south edge (v = -pd) at the bottom up to v = 0 at ph
    for (let i = 0; i < n; i++) {
      const u0 = -total / 2 + i * (pw + 0.08), u1 = u0 + pw;
      const nrm = [0, -Math.sin(tilt), Math.cos(tilt)];
      quad([[u0, -pd, 0.25], [u1, -pd, 0.25], [u1, 0, 0.25 + ph], [u0, 0, 0.25 + ph]], nrm,
        [0.12, 0.15, 0.22]);
      // the aluminium edge underneath, so the panel is not paper-thin from behind
      quad([[u1, -pd, 0.2], [u0, -pd, 0.2], [u0, 0, 0.2 + ph], [u1, 0, 0.2 + ph]],
        [0, Math.sin(tilt), -Math.cos(tilt)], [0.62, 0.63, 0.64]);
    }
    // the tank, behind and above the panels: a long white box with a rounded
    // look from its shading is enough at the distances it is seen from
    const tl = Math.max(total, 1.4), tr = 0.28;
    box(-tl / 2, tl / 2, 0.05, 0.05 + tr * 2, ph + 0.05, ph + 0.05 + tr * 2, [0.86, 0.86, 0.84]);
    // the frame's legs
    box(-tl / 2, -tl / 2 + 0.06, 0.05, 0.11, 0, ph + 0.05, [0.5, 0.5, 0.5]);
    box(tl / 2 - 0.06, tl / 2, 0.05, 0.11, 0, ph + 0.05, [0.5, 0.5, 0.5]);
  }

  /* ---------- shaders ---------- */

  const COMMON = /* glsl */`
    uniform vec3 uSun;
    uniform vec3 uCam;
    uniform vec3 uFog;
    uniform vec3 uSky;
    float h11(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
    float h21(vec2 p) { p = mod(p, 997.0); return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    float vnoise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + vec2(1, 1)), f.x), f.y);
    }
    float fbm(vec2 p) { return 0.55 * vnoise(p) + 0.3 * vnoise(p * 2.03 + 7.1) + 0.15 * vnoise(p * 4.1 + 3.7); }
    // a box from a to b, antialiased by the pixel footprint w
    float rect(vec2 p, vec2 a, vec2 b, vec2 w) {
      vec2 s = smoothstep(a - w, a + w, p) * (1.0 - smoothstep(b - w, b + w, p));
      return s.x * s.y;
    }
    vec3 shade(vec3 base, vec3 n, float ao) {
      float sun = max(dot(n, uSun), 0.0);
      float sky = 0.5 + 0.5 * n.z;
      return base * (0.5 * ao + 0.16 * sky + 0.52 * sun);
    }
    vec3 fogged(vec3 c, vec3 pos) {
      float d = length(pos - uCam);
      return mix(c, uFog, smoothstep(900.0, 5000.0, d) * 0.55);
    }
  `;

  /* The colour and aux of a wall are the same at its four corners, and are
   * `flat`: interpolated, a 3 arrives in some pixels as 2.9999998, the
   * window hash then picks a different window there, and the facade came
   * out sprinkled with pixels of another facade. */
  const WALL_VS = /* glsl */`
    attribute vec4 facade;
    attribute vec3 nrm;
    attribute vec3 col;
    attribute vec4 aux;
    varying vec4 vF;
    varying vec3 vN;
    flat varying vec3 vCol;
    flat varying vec4 vAux;
    varying vec3 vPos;
    void main() {
      vF = facade; vN = nrm; vCol = col; vAux = aux; vPos = position;
      gl_Position = projectionMatrix * vec4(position, 1.0);
    }
  `;

  const WALL_FS = /* glsl */`
    precision highp float;
    ${COMMON}
    varying vec4 vF;
    varying vec3 vN;
    flat varying vec3 vCol;
    flat varying vec4 vAux;
    varying vec3 vPos;

    void main() {
      float u = vF.x, v = vF.y, L = vF.z, H = vF.w;
      float seed = vAux.x;
      float edge = mod(vAux.y, 1000.0);
      bool front = vAux.y >= 1000.0;
      float kind = vAux.z;                 // see the buffers in build()
      bool hipped = vAux.w < -0.5;
      float par = max(vAux.w, 0.0);        // the parapet above the floors
      vec2 uv = vec2(u, v);
      vec2 w = max(fwidth(uv), vec2(1e-4));   // metres per pixel
      float far = smoothstep(0.12, 0.45, max(w.x, w.y));  // detail fades out with distance

      // ---- plaster
      vec2 wp = vec2(u + edge * 17.3 + seed * 113.0, v);
      float grain = fbm(wp * 3.0) - 0.5;
      float stain = fbm(vec2(wp.x * 0.35, wp.y * 0.9) + 31.0);
      vec3 plaster = vCol * (1.0 + grain * 0.07 * (1.0 - far));
      plaster *= 1.0 - 0.10 * smoothstep(0.55, 0.85, stain);        // weathering patches
      // rain splash and dust along the bottom
      float splash = (1.0 - smoothstep(0.0, 0.7, v)) * (0.6 + 0.4 * vnoise(vec2(wp.x * 4.0, 0.0)));
      plaster *= 1.0 - 0.18 * splash;
      vec3 col = plaster;
      float ao = 0.78 + 0.22 * smoothstep(0.0, 1.2, v);              // the ground's shadow
      bool house = kind > 0.5 && kind < 2.5;

      // ---- the garden walls
      if (kind > 2.5) {
        bool top = vN.z > 0.5;
        vec2 q = top ? vPos.xy : uv;
        if (kind < 3.5) {
          // plastered, with a coping a shade lighter along the top
          col = top ? vCol * 1.04 : mix(plaster, vCol * 1.06, smoothstep(H - 0.08 - w.y, H - 0.08 + w.y, v));
        } else if (kind < 4.5) {
          // bare concrete block, 40 by 20, every other course half a block over
          float row = floor(v / 0.2);
          float bx = (u + mod(row, 2.0) * 0.2) / 0.4;
          float dx = min(fract(bx), 1.0 - fract(bx)) * 0.4;
          float dy = min(fract(v / 0.2), 1.0 - fract(v / 0.2)) * 0.2;
          float mortar = 1.0 - smoothstep(0.004, 0.004 + max(w.x, w.y), min(dx, dy));
          float tone = 0.9 + 0.14 * h21(vec2(floor(bx), row) + seed * 30.0);
          col = top ? vCol * 1.03 * (0.95 + 0.1 * grain)
                    : plaster * mix(tone, 1.0, far) * (1.0 - 0.28 * mortar * (1.0 - far));
        } else {
          // a hedge: clumps of leaves, lighter where they catch the light
          float clump = fbm(q * 2.2 + seed * 20.0);
          float leaf = fbm(q * 11.0 + 5.0);
          col = vCol * (0.65 + 0.7 * clump) * mix(0.8 + 0.4 * leaf, 1.0, far);
          ao *= 0.8 + 0.2 * clump;
        }
      }

      // a plinth, on some houses: grey render or stone, the first 40-60 cm
      // a plinth, on some houses: grey render or stone, the first 40-60 cm
      float s1 = h11(seed * 7.13 + 1.0);
      if (house && s1 < 0.55) {
        float ph = 0.4 + 0.2 * h11(seed * 3.7);
        vec3 pc = s1 < 0.25 ? vec3(0.55, 0.54, 0.52) : vec3(0.72, 0.66, 0.56);
        float stone = 0.9 + 0.1 * vnoise(wp * vec2(2.0, 4.0));
        col = mix(col, pc * stone, (1.0 - smoothstep(ph - w.y, ph + w.y, v)));
      }
      // the coping along the top of a flat roof's parapet, and the shadow
      // it throws on the wall under it
      if (par > 0.0) {
        float T = H + par;
        col = mix(col, vCol * 1.08, smoothstep(T - 0.12 - w.y, T - 0.12 + w.y, v));
        float under = smoothstep(T - 0.24, T - 0.14, v) * (1.0 - smoothstep(T - 0.14, T - 0.12, v));
        ao *= 1.0 - 0.3 * under * (1.0 - far);
        // a line where the parapet meets the floors below, drawn or cast
        float band = smoothstep(H - 0.05 - w.y, H - 0.05, v) * (1.0 - smoothstep(H, H + w.y, v));
        col *= 1.0 - 0.12 * band * (1.0 - far) * step(0.5, kind);
      }

      // ---- windows
      float glassCover = 0.0;
      if (house && v < H) {
        float nf = max(1.0, floor(H / 2.95 + 0.25));
        float fh = H / nf;
        float floorI = floor(v / fh);
        float fv = v - floorI * fh;
        float spacing = mix(2.6, 3.8, h11(seed * 5.31 + 2.0));
        if (kind > 1.5) spacing = mix(2.4, 3.2, h11(seed * 5.31 + 2.0));
        float n = floor((L - 0.8) / spacing);
        float ww = mix(0.9, 1.6, h11(seed * 9.7 + 4.0));
        float wh = mix(1.05, 1.45, h11(seed * 2.9 + 5.0));
        float sill = 0.9;
        bool woodShutters = kind < 1.5 && H < 5.5 && h11(seed * 4.1 + 6.0) < 0.4;
        vec3 shutterCol = woodShutters
          ? mix(vec3(0.22, 0.42, 0.33), vec3(0.24, 0.41, 0.55), step(0.5, h11(seed * 8.8)))
          : mix(vec3(0.86, 0.86, 0.84), vec3(0.62, 0.62, 0.6), step(0.6, h11(seed * 8.8)));
        bool balconies = kind > 1.5 && h11(seed * 6.6 + 3.0) < 0.6;
        if (n >= 1.0) {
          float cw = L / n;
          float ci = floor(u / cw);
          float cu = u - (ci + 0.5) * cw;
          float wr = h11(seed * 31.0 + edge * 7.0 + ci * 1.7 + floorI * 13.1);
          bool door = front && floorI < 0.5 && ci == floor(n * 0.5);
          bool skip = !door && kind < 1.5 && wr < 0.12;
          bool balcony = balconies && floorI > 0.5 && mod(ci + floorI * 0.0, 2.0) < 1.0 && n >= 2.0;
          if (door) {
            // a door, wood or painted metal, with a step and a little canopy
            float dw = 1.0, dh = 2.15;
            vec3 dc = h11(seed * 12.1) < 0.5 ? vec3(0.42, 0.28, 0.18) : vec3(0.32, 0.36, 0.38);
            float m = rect(vec2(cu, v), vec2(-dw / 2.0, 0.05), vec2(dw / 2.0, dh), w);
            float frame = m - rect(vec2(cu, v), vec2(-dw / 2.0 + 0.07, 0.05), vec2(dw / 2.0 - 0.07, dh - 0.07), w);
            float panel = rect(vec2(cu, v), vec2(-dw / 2.0 + 0.15, 0.25), vec2(dw / 2.0 - 0.15, dh - 0.25), w)
              - rect(vec2(cu, v), vec2(-dw / 2.0 + 0.2, 0.3), vec2(dw / 2.0 - 0.2, dh - 0.3), w);
            vec3 d = dc * (1.0 - 0.25 * panel) * (1.0 - 0.35 * frame);
            col = mix(col, d, m * (1.0 - far * 0.5));
            float canopy = rect(vec2(cu, v), vec2(-dw / 2.0 - 0.3, dh + 0.15), vec2(dw / 2.0 + 0.3, dh + 0.28), w);
            col = mix(col, vec3(0.5, 0.5, 0.5), canopy * (1.0 - far));
            ao *= 1.0 - 0.3 * rect(vec2(cu, v), vec2(-dw / 2.0 - 0.3, dh - 0.4), vec2(dw / 2.0 + 0.3, dh + 0.15), w) * (1.0 - m);
          } else if (balcony) {
            // a balcony: the wall behind recessed in shadow, a wide opening, a railing
            float bw = min(cw * 0.86, 3.4);
            float open = rect(vec2(cu, fv), vec2(-bw / 2.0, 0.05), vec2(bw / 2.0, fh - 0.45), w);
            vec3 inside = vCol * 0.55;
            float door2 = rect(vec2(cu, fv), vec2(-bw / 2.0 + 0.35, 0.1), vec2(bw / 2.0 - 0.35, 2.1), w);
            inside = mix(inside, vec3(0.16, 0.2, 0.24), door2);
            float slab = rect(vec2(cu, fv), vec2(-bw / 2.0 - 0.1, 0.0), vec2(bw / 2.0 + 0.1, 0.12), w);
            float rail = rect(vec2(cu, fv), vec2(-bw / 2.0, 0.12), vec2(bw / 2.0, 1.05), w);
            float bars = smoothstep(0.35, 0.5, abs(fract(cu / 0.12) - 0.5)) * (1.0 - far);
            float railTop = rect(vec2(cu, fv), vec2(-bw / 2.0, 1.0), vec2(bw / 2.0, 1.07), w);
            vec3 c2 = mix(inside, vec3(0.2, 0.21, 0.22), rail * max(bars * 0.85, far * 0.4));
            c2 = mix(c2, vec3(0.3, 0.3, 0.3), railTop);
            col = mix(col, c2, open);
            col = mix(col, vCol * 0.9, slab);
            ao *= 1.0 - 0.15 * open;
            glassCover = open * door2 * 0.5;
          } else if (!skip) {
            float bot = floorI < 0.5 ? max(sill, 0.95) : sill;
            vec2 a = vec2(-ww / 2.0, bot), b = vec2(ww / 2.0, bot + wh);
            vec2 p = vec2(cu, fv);
            float hole = rect(p, a, b, w);
            float glass = rect(p, a + 0.06, b - 0.06, w);
            // mullion of a two-leaf window
            float mull = rect(p, vec2(-0.03, a.y), vec2(0.03, b.y), w) * step(1.1, ww);
            // the roller shutter: down to a different height in each window
            float down = woodShutters ? 0.0 : pow(h11(wr * 91.0), 1.6) * 0.95;
            float sh = rect(p, vec2(a.x, b.y - down * wh), b, w);
            float slats = 0.85 + 0.15 * smoothstep(0.3, 0.5, abs(fract(fv / 0.05) - 0.5)) * (1.0 - far);
            // inside: dark, or a curtain
            vec3 room = mix(vec3(0.1, 0.12, 0.14), vec3(0.62, 0.56, 0.48), step(0.72, h11(wr * 17.0)) * 0.7);
            // the sky in the glass, more of it at a glancing angle
            vec3 V = normalize(uCam - vPos);
            float fres = pow(1.0 - abs(dot(V, vN)), 3.0);
            vec3 gl = mix(room, uSky * 0.9, 0.25 + 0.55 * fres);
            vec3 frameC = woodShutters ? vec3(0.9, 0.88, 0.84) : vec3(0.78, 0.79, 0.8);
            vec3 win = mix(frameC, gl, glass);
            win = mix(win, frameC, mull);
            win = mix(win, shutterCol * slats, sh);
            // bars over the ground-floor windows of most houses
            if (floorI < 0.5 && h11(seed * 3.3 + 9.0) < 0.6) {
              float bar = 1.0 - smoothstep(0.008, 0.008 + w.x, abs(fract(cu / 0.13 + 0.5) - 0.5) * 0.13);
              float mid = 1.0 - smoothstep(0.01, 0.01 + w.y, abs(fv - (a.y + wh * 0.5)));
              win = mix(win, vec3(0.17, 0.17, 0.18), max(bar, mid) * glass * (1.0 - far));
            }
            // the shutter box above the window
            float boxv = woodShutters ? 0.0 : rect(p, vec2(a.x, b.y), vec2(b.x, b.y + 0.18), w);
            col = mix(col, vCol * 0.93, boxv * (1.0 - far));
            col = mix(col, win, hole);
            glassCover = glass * (1.0 - sh);
            // sill, a little wider and lighter, with its shadow below
            float sl = rect(p, vec2(a.x - 0.06, a.y - 0.06), vec2(b.x + 0.06, a.y), w);
            col = mix(col, vec3(0.86, 0.84, 0.8), sl * (1.0 - far * 0.7));
            ao *= 1.0 - 0.18 * rect(p, vec2(a.x - 0.06, a.y - 0.16), vec2(b.x + 0.06, a.y - 0.06), w) * (1.0 - far);
            // the reveal: the window sits back in the wall, its top in shadow
            ao *= 1.0 - 0.25 * hole * smoothstep(b.y - 0.12, b.y, p.y);
            // wooden shutters, open, on either side
            if (woodShutters) {
              float leaf = rect(p, vec2(a.x - ww * 0.5, a.y), vec2(a.x - 0.02, b.y), w)
                         + rect(p, vec2(b.x + 0.02, a.y), vec2(b.x + ww * 0.5, b.y), w);
              float louv = 0.8 + 0.2 * smoothstep(0.25, 0.5, abs(fract(fv / 0.07) - 0.5)) * (1.0 - far);
              col = mix(col, shutterCol * louv, leaf);
            }
            // a dark streak of dirt washing down from the sill
            float streak = (1.0 - smoothstep(0.0, 1.4, a.y - p.y)) * step(p.y, a.y) * step(abs(cu), ww * 0.45);
            col *= 1.0 - 0.06 * streak * vnoise(vec2(cu * 9.0, 0.0)) * (1.0 - far);
            // an air conditioner beside some windows, its shadow, and the
            // stain its drip has left on the wall below
            float acx0 = b.x + 0.2, acx1 = b.x + 1.0;
            if (!woodShutters && h11(wr * 53.0) < 0.28 && acx1 < cw * 0.5 - 0.1) {
              vec2 q0 = vec2(acx0, a.y - 0.05), q1 = vec2(acx1, a.y + 0.5);
              float ac = rect(p, q0, q1, w);
              vec2 fc = vec2(acx0 + 0.27, a.y + 0.22);
              float r = length(p - fc);
              float fan = 1.0 - smoothstep(0.17, 0.17 + w.x, r);
              float grille = 0.8 + 0.2 * smoothstep(0.3, 0.5, abs(fract(r / 0.03) - 0.5));
              vec3 acc = vec3(0.86, 0.86, 0.84) * mix(1.0, 0.5 * grille, fan * (1.0 - far));
              ao *= 1.0 - 0.3 * rect(p, vec2(acx0, q0.y - 0.12), vec2(acx1, q0.y), w);
              float drip = step(p.y, q0.y) * (1.0 - smoothstep(0.0, 2.5, q0.y - p.y))
                         * (1.0 - smoothstep(0.03, 0.12, abs(p.x - acx1 + 0.08)));
              col *= 1.0 - 0.2 * drip * (1.0 - far);
              col = mix(col, acc, ac);
            }
          }
        }
      }

      vec3 c = shade(col, normalize(vN), ao);
      // the glass is lit by the sky, not the sun
      c = mix(c, col * 0.95, glassCover * 0.6);
      gl_FragColor = vec4(fogged(c, vPos), 1.0);
      #include <colorspace_fragment>
    }
  `;

  const ROOF_VS = /* glsl */`
    attribute vec3 nrm3;
    attribute vec3 col;
    attribute vec4 aux;
    varying vec3 vN;
    flat varying vec3 vCol;
    flat varying vec4 vAux;
    varying vec3 vPos;
    void main() {
      vN = nrm3; vCol = col; vAux = aux; vPos = position;
      gl_Position = projectionMatrix * vec4(position, 1.0);
    }
  `;

  const ROOF_FS = /* glsl */`
    precision highp float;
    ${COMMON}
    varying vec3 vN;
    flat varying vec3 vCol;
    flat varying vec4 vAux;
    varying vec3 vPos;
    void main() {
      vec2 uv = vAux.zw;
      vec2 w = max(fwidth(uv), vec2(1e-4));
      float far = smoothstep(0.08, 0.3, max(w.x, w.y));
      vec3 c = vCol;
      if (vAux.y > 0.5) {
        // clay tiles: courses 32 cm apart, each tile 22 cm wide, every other
        // course shifted half a tile, each tile its own shade of red
        float row = floor(uv.y / 0.32);
        float x = uv.x / 0.22 + mod(row, 2.0) * 0.5;
        float ti = floor(x);
        float tv = fract(uv.y / 0.32);
        float tone = 0.85 + 0.25 * h21(vec2(ti, row) + vAux.x * 50.0);
        float roll = 0.85 + 0.15 * sin(fract(x) * 3.14159);
        float lip = 1.0 - 0.35 * smoothstep(0.8, 1.0, tv);
        float grime = 0.9 + 0.1 * fbm(uv * 0.8 + vAux.x * 9.0);
        vec3 tc = c * tone * mix(roll * lip, 0.9, far) * grime;
        c = tc;
      } else {
        // a flat roof: grey or white waterproofing, patched, with its stains
        float g = fbm(uv * 1.3 + vAux.x * 40.0);
        float patchy = smoothstep(0.55, 0.75, fbm(uv * 0.25 + 11.0 + vAux.x * 9.0));
        c *= 0.92 + 0.12 * g;
        c *= 1.0 - 0.1 * patchy;
      }
      vec3 n = normalize(vN);
      gl_FragColor = vec4(fogged(shade(c, n, 1.0), vPos), 1.0);
      #include <colorspace_fragment>
    }
  `;

  const LIT_VS = /* glsl */`
    attribute vec3 nrm3;
    attribute vec3 col;
    varying vec3 vN;
    flat varying vec3 vCol;
    varying vec3 vPos;
    void main() {
      vN = nrm3; vCol = col; vPos = position;
      gl_Position = projectionMatrix * vec4(position, 1.0);
    }
  `;

  const LIT_FS = /* glsl */`
    precision highp float;
    ${COMMON}
    varying vec3 vN;
    flat varying vec3 vCol;
    varying vec3 vPos;
    void main() {
      vec3 n = normalize(vN);
      vec3 c = shade(vCol, n, 1.0);
      // the collectors catch the sky
      if (vCol.b > vCol.r + 0.05) {
        vec3 V = normalize(uCam - vPos);
        c = mix(c, uSky * 0.7, 0.35 * pow(1.0 - abs(dot(V, n)), 2.0) + 0.1);
      }
      gl_FragColor = vec4(fogged(c, vPos), 1.0);
      #include <colorspace_fragment>
    }
  `;

  /* ---------- the layer ---------- */

  async function ready(raw, rawWalls) {
    if (built) return built;
    if (building) return building;
    building = (async () => {
      THREE = await import(THREE_URL);
      const t0 = performance.now();
      built = await build(raw, rawWalls);
      built.ms = Math.round(performance.now() - t0);
      return built;
    })();
    try {
      return await building;
    } catch (err) {
      building = null;
      throw err;
    }
  }

  const camera = { m: null };

  function layer(map) {
    return {
      id: 'm3d-facades',
      type: 'custom',
      renderingMode: '3d',
      onAdd(m, gl) {
        if (!renderer) {
          renderer = new THREE.WebGLRenderer({ canvas: m.getCanvas(), context: gl, antialias: true });
          renderer.autoClear = false;
          renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
        }
        if (!camera.m) {
          camera.m = new THREE.Camera();
          camera.m.matrixAutoUpdate = false;
        }
      },
      render(gl, matrix) {
        if (!built) return;
        const { ox, oy, s0 } = built;
        const M = new THREE.Matrix4().fromArray(matrix);
        const l = new THREE.Matrix4().makeTranslation(ox, oy, 0)
          .scale(new THREE.Vector3(s0, -s0, s0));
        const P = M.multiply(l);
        camera.m.projectionMatrix = P;
        camera.m.projectionMatrixInverse.copy(P).invert();
        // the eye, in model units: the point the projection sends to infinity
        const e = new THREE.Vector4(0, 0, 1, 0).applyMatrix4(camera.m.projectionMatrixInverse);
        if (Math.abs(e.w) > 1e-12) built.uni.uCam.value.set(e.x / e.w, e.y / e.w, e.z / e.w);
        renderer.resetState();
        renderer.render(built.scene, camera.m);
      }
    };
  }

  /** Put the faces on the map, over the plain extrusions, and hide those
   *  once the faces are drawn. `raw` and `rawWalls` are buildings.json and
   *  walls.json as they came. */
  async function attach(map, raw, rawWalls, before) {
    try {
      await ready(raw, rawWalls);
    } catch (err) {
      console.warn('facades: Three.js did not load; the houses stay plain', err);
      return false;
    }
    if (!map.getStyle() || !map.getSource('m3d-houses')) return false;
    if (!map.getLayer('m3d-facades')) map.addLayer(layer(map), before);
    for (const id of ['m3d-houses', 'm3d-roofs', 'm3d-walls']) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', 'none');
    }
    map.triggerRepaint();
    return true;
  }

  return {
    attach,
    stats: () => built && { buildings: built.count, triangles: built.tris, ms: built.ms }
  };
})();
