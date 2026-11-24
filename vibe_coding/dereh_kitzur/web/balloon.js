/* The balloon you stand in (24/11/2026).
 *
 * Until now the balloon was a flight model and a sound: the picture was the
 * same photograph of the moshava the jet sees, with a brown strip along the
 * bottom for a basket. Ori asked for the thing itself - the burner, some of
 * the envelope, some of the basket, sandbags on the side - so that a person
 * "really feels they are in a balloon". This file is that: a wicker basket
 * with a leather rim, four padded uprights, a stainless burner frame with a
 * double burner and its flames, the mouth of the envelope and the whole
 * inside of it above, the flying wires, the red line to the valve at the
 * crown, and canvas sandbags hung on the outside. Drawn by Three.js on a
 * transparent canvas over the map, the way the bike is.
 *
 * It is a view from where you stand, and it is not the map's camera. The map
 * looks at the town through a 32 degree window, which is a long lens: through
 * a lens like that a basket is one blurred post. What a person in a basket
 * actually has is peripheral vision - the rim under their hands, the frame
 * over their head - and what a camera in a basket has is a very wide lens.
 * So the rig is drawn through a wide lens of its own (see LENS), set a
 * little upward, and laid over the map's picture, the way a first-person
 * game draws the hands and the world with two different cameras. Nothing in
 * the basket has to line up with anything on the ground, and what would give
 * the trick away - the horizon - is not something a basket has.
 *
 * What ties the two pictures together is everything else:
 *
 *   - the light. The sun is where a dawn sun is, low in the east, and the
 *     rig turns under it as the basket turns, so the lit side of an upright
 *     moves round it when you face somewhere else;
 *   - the burner. Its flame is the same `burn` the sound follows, it lights
 *     the frame, the uprights, the rim and the inside of the envelope, and
 *     it flickers;
 *   - the swing. The basket is the pendulum explore.js already integrates.
 *     Your head does not follow it rigidly, so the rig shifts and tips a
 *     little against the picture, and everything that hangs - the sandbags,
 *     the valve line - hangs from the true vertical and swings on its own
 *     shorter period;
 *   - where you look. W leans you out over the rim, and the head goes with
 *     it: the frame leaves the picture, the outside of the basket and the
 *     bags hanging on it come in. S tips the head back, and you are looking
 *     up the flame into the envelope, all twenty metres of it, to the valve
 *     at the crown, which opens when the down arrow is held.
 *
 * Everything is made here: no model files, no image files. The textures are
 * drawn into canvases at load (a weave has a height, the height gives the
 * normal map and the shadow in the gaps), which is why the wicker catches
 * the light the way wicker does instead of looking like a photograph of it.
 *
 * Sandbags, honestly: a hot-air balloon does not carry them - its ballast is
 * the heat, and that is the whole flight model. They belong to gas balloons.
 * Ori asked for them because they are what a balloon basket looks like in
 * every picture anyone has seen, and he is right, so they hang here.
 *
 * explore.js owns the flight and calls update() and render() every frame;
 * this file only draws. Three.js comes from the same address bike.js uses,
 * so whichever flies first pays for it and the other finds it loaded.
 */
'use strict';

const Balloon = (() => {

  const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
  let THREE = null;
  let loading = null;

  /** Fetch three.js, once. A failure is forgotten so the next take-off can
   *  try again; until then the flat basket in app.css is what is shown. */
  function load() {
    if (!loading) {
      loading = import(THREE_URL).then((m) => { THREE = m; return true; });
      loading.catch(() => { loading = null; });
    }
    return loading;
  }

  const RAD = Math.PI / 180;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smooth = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };

  /* ---------- the rig's measurements, metres ----------
   *
   * A four-person basket and an AX8 envelope, the one the flight model is.
   * Origin on the basket floor at its middle; y up; you face -z, +x is your
   * right. */
  const G = {
    W: 1.72, D: 1.08,         // basket, outside, across and front to back
    H: 1.12,                  // floor to the top of the wicker
    CORNER: 0.13,             // radius of the basket's corners
    WALL: 0.05,               // thickness of the weave
    RIM: 0.058,               // radius of the padded rim
    FRAME_Y: 2.24,            // the burner frame
    FRAME_W: 1.24, FRAME_D: 0.64,
    BURNER_X: 0.135,          // the two burners, either side of the middle
    COIL_TOP: 2.68,           // where the flame leaves
    MOUTH_Y: 4.75, MOUTH_R: 1.95,
    SKIRT_Y: 4.02, SKIRT_R: 1.78,
    GORES: 24,
    TOP_Y: 25.0               // the crown
  };

  /* ---------- the lens ----------
   *
   * Where the head is and what it looks through. The eye stands a little to
   * the right of the middle and back from it, so the burner is up and to the
   * left rather than hanging over the middle of the picture, the near
   * upright is near and the far one far. On a tall phone screen the lens
   * cannot be as wide across, so the eye moves back to the middle. */
  const LENS = {
    VFOV: 102,                // degrees, at least; see fitLens
    HFOV_MIN: 62,             // a tall screen is given this much across, by widening the vertical
    VFOV_MAX: 108,
    EYE: [0.0, 1.58, 0.36],
    EYE_TALL: [0.0, 1.58, 0.36],
    /* A tall screen is narrow, and two burners half a metre over your head
     * are the whole width of it: a picture of the underside of a valve
     * block. So there the frame, the burners and the envelope are set half
     * a metre higher and a little forward of where they are - the same
     * things at the same angle, at twice the distance and half the size.
     * The uprights follow, and are out of a tall picture anyway. */
    TALL: [0.50, -0.35],      // up, and towards the far wall: metres
    PITCH: 15,                // the head is tipped back this much from level: degrees
    YAW: 0,                   // and turned this much towards the middle of the basket
    UP: 34,                   // S: this much more,
    VFOV_UP: 84,              // through a longer lens, so the uprights behind you stay out of it
    GLANCE: 17,               // and the burner lit: you look up at a flame you have just opened
    GLANCE_TALL: 9,           // less on a tall screen, where the burners are the whole width of it
    GLANCE_IN: 0.45,          // seconds for the eyes to go up to it,
    GLANCE_OUT: 1.1,          // and to come back down once it is shut
    DOWN: 46,                 // W: this much the other way, leaning out
    LEAN: [0.0, 1.63, -0.33],   // where the head is when it is over the rim
    ROLL_K: 0.30,             // the share of the basket's swing the head does not follow
    PITCH_K: 0.35
  };

  /* ---------- small tools ---------- */

  /** A seeded random stream, so the textures are the same on every load. */
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Value noise that repeats every `px`,`py` cells, for textures that tile. */
  function tileNoise(seed, px, py) {
    const r = rng(seed);
    const g = new Float32Array(px * py);
    for (let i = 0; i < g.length; i++) g[i] = r();
    return (x, y) => {
      const xi = Math.floor(x), yi = Math.floor(y);
      const fx = x - xi, fy = y - yi;
      const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
      const x0 = ((xi % px) + px) % px, x1 = (x0 + 1) % px;
      const y0 = ((yi % py) + py) % py, y1 = (y0 + 1) % py;
      const a = g[y0 * px + x0], b = g[y0 * px + x1], c = g[y1 * px + x0], d = g[y1 * px + x1];
      return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
    };
  }

  function makeCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }

  /** A normal map from a height field that wraps. `k` is how steep. */
  function normalsFrom(H, w, h, k) {
    const c = makeCanvas(w, h);
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(w, h);
    const d = img.data;
    for (let y = 0; y < h; y++) {
      const yu = (y + h - 1) % h, yd = (y + 1) % h;
      for (let x = 0; x < w; x++) {
        const xl = (x + w - 1) % w, xr = (x + 1) % w;
        const dx = (H[y * w + xr] - H[y * w + xl]) * k;
        const dy = (H[yd * w + x] - H[yu * w + x]) * k;   // down the image, which is down the surface
        const inv = 1 / Math.hypot(dx, dy, 1);
        const i = (y * w + x) * 4;
        d[i] = (-dx * inv * 0.5 + 0.5) * 255;
        d[i + 1] = (dy * inv * 0.5 + 0.5) * 255;
        d[i + 2] = (inv * 0.5 + 0.5) * 255;
        d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  function tex(canvas, { srgb = false, rx = 1, ry = 1, aniso = 8 } = {}) {
    const t = new THREE.CanvasTexture(canvas);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(rx, ry);
    t.anisotropy = aniso;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  /* ---------- textures ----------
   *
   * Each is a height field first. The colour is the material's own colour
   * darkened where the height is low, and the normal map is the slope of the
   * same field, so the two cannot disagree. */

  /** Woven willow: upright stakes, and weavers passing in front of one and
   *  behind the next, each row starting on the other foot. */
  function wicker() {
    const S = 512, STAKES = 8, ROWS = 32;
    const sp = S / STAKES, rh = S / ROWS;
    const H = new Float32Array(S * S);
    const c = makeCanvas(S, S);
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(S, S);
    const d = img.data;
    const r = rng(11);
    const rowTone = [], rowHue = [];
    for (let i = 0; i < ROWS; i++) {
      rowTone.push(0.86 + 0.24 * r() - (r() < 0.12 ? 0.16 : 0));
      rowHue.push(r());
    }
    const segTone = [];
    for (let i = 0; i < ROWS * STAKES; i++) segTone.push(0.9 + 0.2 * r());
    const fibre = tileNoise(5, 16, 128);
    const blotch = tileNoise(9, 6, 6);
    for (let y = 0; y < S; y++) {
      const row = Math.floor(y / rh);
      const fy = (y + 0.5) / rh - row - 0.5;
      for (let x = 0; x < S; x++) {
        const sx = (x + 0.5) / sp;
        const col = Math.round(sx);
        const dx = (sx - col) * sp;
        const u = Math.cos(Math.PI * (sx + row));            // +1 where this row passes in front of the stake
        const pw = Math.sqrt(Math.max(0, 1 - (fy / 0.5) * (fy / 0.5)));
        const hw = 0.40 + 0.34 * pw + 0.22 * u;
        const SW = 7.5;
        const ps = Math.abs(dx) < SW ? Math.sqrt(1 - (dx / SW) * (dx / SW)) : -1;
        const hs = ps < 0 ? -1 : 0.40 + 0.30 * ps - 0.26 * u;
        const weaver = hw >= hs;
        const h = Math.max(hw, hs) * (0.55 + 0.45 * Math.min(1, pw * 1.6));
        H[y * S + x] = h;
        const shade = 0.30 + 0.70 * smooth((h - 0.22) / 0.62);
        const f = 0.86 + 0.28 * fibre(x / 32, y / 4);
        const b = 0.90 + 0.20 * blotch(x / 85.3, y / 85.3);
        let R, Gc, B;
        if (weaver) {
          const seg = segTone[row * STAKES + (Math.floor(sx + (row & 1 ? 0.5 : 0)) % STAKES)];
          const t = rowTone[row] * seg * f * b * shade;
          const hue = rowHue[row];
          R = (188 + 14 * hue) * t; Gc = (140 + 6 * hue) * t; B = (80 - 10 * hue) * t;
        } else {
          const t = f * b * shade * 0.92;
          R = 160 * t; Gc = 112 * t; B = 62 * t;
        }
        const i = (y * S + x) * 4;
        d[i] = R; d[i + 1] = Gc; d[i + 2] = B; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { map: c, normal: normalsFrom(H, S, S, 5.5) };
  }

  /** Leather for the rim and the padded sleeves of the uprights: a fine
   *  pebbled grain, worn paler where hands go, with a line of stitching. */
  function leather(seed, base, stitch) {
    const S = 256;
    const H = new Float32Array(S * S);
    const c = makeCanvas(S, S);
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(S, S);
    const d = img.data;
    const fine = tileNoise(seed, 64, 64), mid = tileNoise(seed + 1, 16, 16), wide = tileNoise(seed + 2, 4, 4);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const n = 0.55 * fine(x / 4, y / 4) + 0.3 * mid(x / 16, y / 16) + 0.15 * wide(x / 64, y / 64);
        let h = n;
        let t = 0.84 + 0.30 * mid(x / 16, y / 16) * wide(x / 64, y / 64) + 0.16 * (fine(x / 4, y / 4) - 0.5);
        let R = base[0] * t, Gc = base[1] * t, B = base[2] * t;
        if (stitch) {
          // The seam: a groove, and thread crossing it every few millimetres.
          const sy = Math.abs(y - S * 0.5);
          if (sy < 5) {
            h -= 0.35 * (1 - sy / 5);
            const k = 0.72 + 0.28 * (sy / 5);
            R *= k; Gc *= k; B *= k;
          }
          if (sy < 9 && (x % 16) < 9 && sy > 1.5) {
            const on = Math.abs(sy - 5.5) < 2.2;
            if (on) { R = 214; Gc = 196; B = 160; h += 0.3; }
          }
        }
        H[y * S + x] = h;
        const i = (y * S + x) * 4;
        d[i] = R; d[i + 1] = Gc; d[i + 2] = B; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { map: c, normal: normalsFrom(H, S, S, 2.6) };
  }

  /** Sacking: a coarse plain weave, with the weight stencilled on it the way
   *  ballast bags are marked. */
  function sacking() {
    const S = 256;
    const H = new Float32Array(S * S);
    const c = makeCanvas(S, S);
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(S, S);
    const d = img.data;
    const stain = tileNoise(21, 5, 5), slub = tileNoise(22, 64, 64);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const P = 2;
        const cx = Math.floor(x / P), cy = Math.floor(y / P);
        const fx = (x % P) / P - 0.5, fy = (y % P) / P - 0.5;
        const over = (cx + cy) % 2 === 0;       // warp on top here, weft there
        const across = over ? fx : fy;
        const h = Math.sqrt(Math.max(0, 1 - across * across * 3.4)) * (over ? 1 : 0.86)
                + 0.25 * (slub(x / 4, y / 4) - 0.5);
        H[y * S + x] = h;
        const t = (0.80 + 0.22 * h) * (0.82 + 0.32 * stain(x / 51.2, y / 51.2));
        const i = (y * S + x) * 4;
        d[i] = 204 * t; d[i + 1] = 190 * t; d[i + 2] = 160 * t; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    // The stencil, worn: drawn pale-dark and then rubbed out in specks.
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = '#3b2c1c';
    ctx.font = '700 46px "DejaVu Sans Mono", "Courier New", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('15 KG', S * 0.5, S * 0.56);
    ctx.restore();
    return { map: c, normal: normalsFrom(H, S, S, 1.6) };
  }

  /** Three-strand rope, laid: diagonal bulges along the length. */
  function rope(base) {
    const S = 64;
    const H = new Float32Array(S * S);
    const c = makeCanvas(S, S);
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(S, S);
    const d = img.data;
    const fz = tileNoise(31, 32, 32);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const ph = ((x + y) / S * 4) % 1;
        const h = Math.sin(ph * Math.PI);
        H[y * S + x] = h;
        const t = (0.5 + 0.6 * h) * (0.88 + 0.24 * fz(x / 2, y / 2));
        const i = (y * S + x) * 4;
        d[i] = base[0] * t; d[i + 1] = base[1] * t; d[i + 2] = base[2] * t; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return { map: c, normal: normalsFrom(H, S, S, 3) };
  }

  /** Perforated sheet: rows of round holes, staggered, for the burner cans. */
  function perforated() {
    const S = 64;
    const c = makeCanvas(S, S);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, S, S);
    ctx.fillStyle = '#17130f';
    for (const [x, y] of [[16, 16], [48, 16], [0, 48], [32, 48], [64, 48]]) {
      ctx.beginPath();
      ctx.arc(x, y, 9.5, 0, Math.PI * 2);
      ctx.fill();
    }
    return c;
  }

  /* The envelope's colours. The app's own - the yellow of its buttons and the
   * green of its roundel, the red the icon's skirt has - laid as a spiral,
   * which is the pattern that reads as "balloon" from underneath: each gore's
   * panels are the neighbouring gore's, one step up. */
  const FABRIC = [
    [244, 194, 22],    // the app's yellow, a shade off the screen's
    [244, 194, 22],
    [234, 128, 34],    // orange
    [192, 56, 40],     // red
    [244, 194, 22],
    [244, 194, 22],
    [240, 228, 202],   // cream
    [16, 132, 94]      // the app's green
  ];
  const PANELS = 11;
  const SKIRT_COL = [112, 32, 27];

  /** The envelope's cloth: eight gores of the spiral (it repeats three times
   *  round), every panel with its seams, the load tapes down the gore lines,
   *  and the scoop at the bottom in its own colour. `vMouth` is how far up
   *  the texture the mouth is. */
  function fabric(vMouth) {
    const W = 1024, Hh = 2048, NG = FABRIC.length;
    const gw = W / NG;
    const c = makeCanvas(W, Hh);
    const ctx = c.getContext('2d');
    const r = rng(41);
    const yMouth = Hh * (1 - vMouth);
    const ph = yMouth / PANELS;
    for (let g = 0; g < NG; g++) {
      for (let p = 0; p < PANELS; p++) {
        const col = FABRIC[(g + p) % NG];
        const j = 0.94 + 0.08 * r();          // no two bolts of cloth are quite one colour
        ctx.fillStyle = `rgb(${col[0] * j | 0},${col[1] * j | 0},${col[2] * j | 0})`;
        ctx.fillRect(g * gw, yMouth - (p + 1) * ph, gw + 1, ph + 1);
      }
    }
    // the scoop, Nomex, below the mouth
    ctx.fillStyle = `rgb(${SKIRT_COL.join(',')})`;
    ctx.fillRect(0, yMouth, W, Hh - yMouth);
    // ripstop: a faint grid in the weave
    ctx.fillStyle = 'rgba(0,0,0,.05)';
    for (let x = 0; x < W; x += 8) ctx.fillRect(x, 0, 1, Hh);
    for (let y = 0; y < Hh; y += 8) ctx.fillRect(0, y, W, 1);
    // panel seams, with their double row of stitching
    for (let p = 0; p <= PANELS; p++) {
      const y = yMouth - p * ph;
      ctx.fillStyle = 'rgba(40,20,10,.34)';
      ctx.fillRect(0, y - 2, W, 4);
      ctx.fillStyle = 'rgba(255,255,255,.22)';
      ctx.fillRect(0, y - 4, W, 1);
      ctx.fillRect(0, y + 3, W, 1);
    }
    // load tapes on the gore lines: these carry the basket, and are opaque
    for (let g = 0; g <= NG; g++) {
      const x = g * gw;
      ctx.fillStyle = 'rgba(28,20,16,.78)';
      ctx.fillRect(x - 4, 0, 8, Hh);
      ctx.fillStyle = 'rgba(255,255,255,.16)';
      ctx.fillRect(x - 5, 0, 1, Hh);
      ctx.fillRect(x + 4, 0, 1, Hh);
    }
    // the mouth's own tape
    ctx.fillStyle = 'rgba(24,18,14,.85)';
    ctx.fillRect(0, yMouth - 5, W, 10);
    return c;
  }

  /* ---------- geometry ---------- */

  /** A rounded rectangle in the floor plane, as points going round, with
   *  the outward normal and the distance travelled at each. */
  function loop(w, d, rad, perArc = 10, perSide = 10) {
    const pts = [];
    const hx = w / 2 - rad, hz = d / 2 - rad;
    const corners = [[hx, hz, 0], [-hx, hz, 90], [-hx, -hz, 180], [hx, -hz, 270]];
    const push = (x, z, nx, nz) => pts.push({ x, z, nx, nz });
    for (let k = 0; k < 4; k++) {
      const [cx, cz, a0] = corners[k];
      for (let i = 0; i <= perArc; i++) {
        const a = (a0 + 90 * i / perArc) * RAD;
        const nx = Math.cos(a), nz = Math.sin(a);
        push(cx + nx * rad, cz + nz * rad, nx, nz);
      }
      // the straight after this corner, up to the start of the next
      const [nxc, nzc, na0] = corners[(k + 1) % 4];
      const a1 = (a0 + 90) * RAD, sxn = Math.cos(a1), szn = Math.sin(a1);
      const from = pts[pts.length - 1];
      const to = { x: nxc + Math.cos(na0 * RAD) * rad, z: nzc + Math.sin(na0 * RAD) * rad };
      for (let i = 1; i < perSide; i++) {
        const t = i / perSide;
        push(lerp(from.x, to.x, t), lerp(from.z, to.z, t), sxn, szn);
      }
    }
    let s = 0;
    pts[0].s = 0;
    for (let i = 1; i < pts.length; i++) {
      s += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z);
      pts[i].s = s;
    }
    const total = s + Math.hypot(pts[0].x - pts[pts.length - 1].x, pts[0].z - pts[pts.length - 1].z);
    return { pts, total };
  }

  /** A wall standing on a loop, from y0 to y1, facing out or in. */
  function wallGeo(lp, y0, y1, tile, inward, rows = 6) {
    const n = lp.pts.length;
    const pos = [], nor = [], uv = [], idx = [];
    // The weave has to close on itself: round the tile up so a whole number
    // of them goes round.
    const reps = Math.max(1, Math.round(lp.total / tile));
    for (let i = 0; i <= n; i++) {
      const p = lp.pts[i % n];
      const s = i === n ? lp.total : p.s;
      for (let j = 0; j <= rows; j++) {
        const y = lerp(y0, y1, j / rows);
        pos.push(p.x, y, p.z);
        nor.push(inward ? -p.nx : p.nx, 0, inward ? -p.nz : p.nz);
        uv.push(s / lp.total * reps, y / tile);
      }
    }
    const R = rows + 1;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < rows; j++) {
        const a = i * R + j, b = (i + 1) * R + j, c = a + 1, e = b + 1;
        if (inward) idx.push(a, b, c, c, b, e); else idx.push(a, c, b, c, e, b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    return g;
  }

  /** A closed curve through a loop at a height, for the rim and the frame. */
  function loopCurve(lp, y) {
    const v = lp.pts.map((p) => new THREE.Vector3(p.x, y, p.z));
    return new THREE.CatmullRomCurve3(v, true, 'centripetal');
  }

  /** A rod from a to b. */
  function rod(a, b, r, mat, radial = 10, heightSeg = 1) {
    const dir = new THREE.Vector3().subVectors(b, a);
    const len = dir.length();
    const g = new THREE.CylinderGeometry(r, r, len, radial, heightSeg, false);
    const m = new THREE.Mesh(g, mat);
    m.position.copy(a).addScaledVector(dir, 0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    return m;
  }

  /** The envelope's outline, mouth to crown: radius against height. Drawn by
   *  eye from photographs of an AX8, which is what the flight model is: a
   *  straight cone out of the mouth, widest two thirds of the way up, a
   *  flattened dome. */
  function envelopeProfile() {
    const m = G.MOUTH_Y;
    const ctl = [
      [G.SKIRT_R, G.SKIRT_Y], [G.MOUTH_R, m], [3.35, m + 2.0], [5.15, m + 4.5], [6.9, m + 7.2],
      [8.2, m + 10.0], [8.9, m + 12.7], [8.65, m + 15.2], [7.5, m + 17.3], [5.3, m + 19.0],
      [2.9, m + 19.95], [1.55, m + 20.22]
    ];
    const curve = new THREE.CatmullRomCurve3(ctl.map(([r, y]) => new THREE.Vector3(r, y, 0)), false, 'centripetal');
    const N = 72;
    const pts = curve.getSpacedPoints(N);
    // how far along the mouth is, by the same spacing
    let best = 0;
    for (let i = 0; i <= N; i++) if (Math.abs(pts[i].y - m) < Math.abs(pts[best].y - m)) best = i;
    return { pts, vMouth: best / N };
  }

  /** The envelope: a surface of revolution with each gore bulging between
   *  its load tapes, the way cloth under pressure does. */
  function envelopeGeo(profile) {
    const { pts, vMouth } = profile;
    const NV = pts.length - 1;
    const PER = 8;                             // facets across one gore
    const NA = G.GORES * PER;
    const pos = new Float32Array((NA + 1) * (NV + 1) * 3);
    const uv = new Float32Array((NA + 1) * (NV + 1) * 2);
    const idx = [];
    for (let j = 0; j <= NV; j++) {
      const v = j / NV;
      const up = clamp((v - vMouth) / (1 - vMouth), 0, 1);
      const bulge = 0.050 * Math.pow(Math.sin(Math.PI * Math.pow(up, 0.8)), 0.6);
      for (let i = 0; i <= NA; i++) {
        const a = (i % NA) / NA * Math.PI * 2;
        const across = (i % PER) / PER;
        const r = pts[j].x * (1 + bulge * (Math.sin(Math.PI * across) - 0.64));
        const k = (j * (NA + 1) + i) * 3;
        pos[k] = Math.cos(a) * r;
        pos[k + 1] = pts[j].y;
        pos[k + 2] = Math.sin(a) * r;
        const q = (j * (NA + 1) + i) * 2;
        uv[q] = i / PER / FABRIC.length;       // one texture across eight gores
        uv[q + 1] = v;
      }
    }
    for (let j = 0; j < NV; j++) {
      for (let i = 0; i < NA; i++) {
        const a = j * (NA + 1) + i, b = a + 1, c = a + NA + 1, e = c + 1;
        idx.push(a, c, b, b, c, e);            // front faces look outward
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    // The seam at i = 0 and i = NA is one line of cloth, not an edge.
    const n = g.attributes.normal;
    for (let j = 0; j <= NV; j++) {
      const a = j * (NA + 1), b = a + NA;
      const x = n.getX(a) + n.getX(b), y = n.getY(a) + n.getY(b), z = n.getZ(a) + n.getZ(b);
      const inv = 1 / Math.hypot(x, y, z);
      n.setXYZ(a, x * inv, y * inv, z * inv);
      n.setXYZ(b, x * inv, y * inv, z * inv);
    }
    return g;
  }

  /* ---------- shaders ---------- */

  const NOISE_GLSL = `
    float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
    float vnoise(vec3 x) {
      vec3 i = floor(x), f = fract(x);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x), mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
                 mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x), mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
    }
    float fbm(vec3 p) { return 0.55 * vnoise(p) + 0.30 * vnoise(p * 2.07 + 3.1) + 0.15 * vnoise(p * 4.3 + 7.7); }
  `;

  /* The envelope's cloth is thin, and that is how it is lit: from the inside
   * you see the daylight coming through it, in its own colours, brightest on
   * the side the sun is on; from the outside, underneath, you see cloth in
   * the shade of itself. The burner adds its own light to the inside, most
   * near the mouth and up the line of the flame. */
  function envelopeMaterial(map) {
    return new THREE.ShaderMaterial({
      side: THREE.DoubleSide,
      uniforms: {
        map: { value: map },
        uSun: { value: new THREE.Vector3(1, 0.2, 0) },
        uSunCol: { value: new THREE.Color(1.0, 0.80, 0.58) },
        uSky: { value: new THREE.Color(0.62, 0.72, 0.90) },
        uGround: { value: new THREE.Color(0.34, 0.31, 0.24) },
        uFlame: { value: new THREE.Color(1.0, 0.52, 0.18) },
        uBurn: { value: 0 },
        uTime: { value: 0 },
        uMouthY: { value: G.MOUTH_Y },
        uTopY: { value: G.TOP_Y }
      },
      vertexShader: `
        uniform float uTime, uMouthY;
        varying vec2 vUv; varying vec3 vN; varying vec3 vP;
        void main() {
          vUv = uv;
          vec3 p = position;
          // The scoop is loose cloth and the air under it is not still.
          float loose = clamp((uMouthY - p.y) / 0.7, 0.0, 1.0);
          float a = atan(p.z, p.x);
          p.xz *= 1.0 + loose * (0.022 * sin(a * 9.0 + uTime * 2.3) + 0.014 * sin(a * 17.0 - uTime * 3.7));
          vN = normal;
          vP = p;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }`,
      fragmentShader: `
        uniform sampler2D map;
        uniform vec3 uSun, uSunCol, uSky, uGround, uFlame;
        uniform float uBurn, uMouthY, uTopY;
        varying vec2 vUv; varying vec3 vN; varying vec3 vP;
        void main() {
          vec3 base = texture2D(map, vUv).rgb;
          vec3 n = normalize(vN);
          float sun = dot(n, uSun);
          float up = clamp((vP.y - uMouthY) / (uTopY - uMouthY), 0.0, 1.0);
          vec3 col;
          if (gl_FrontFacing) {
            // outside, seen from below: skylight and what the ground throws back
            vec3 amb = mix(uGround, uSky, n.y * 0.5 + 0.5);
            col = base * (amb * 0.62 + uSunCol * max(sun, 0.0) * 1.25);
            // and the burner's light, coming through from within
            col += base * base * uFlame * uBurn * 0.55 * (1.0 - smoothstep(0.0, 0.5, up));
          } else {
            // inside: the day, through the cloth. Cloth passes its own colour
            // twice over, so what comes through is deeper than the cloth.
            vec3 through = base * mix(vec3(1.0), base, 0.45);
            float lit = 0.25 + 0.52 * max(sun, 0.0) + 0.08 * max(-sun, 0.0);
            // the cloth is in folds of its own shade lower down, and tight and bright above
            lit *= 0.72 + 0.28 * smoothstep(0.0, 0.35, up);
            col = through * mix(uSky, uSunCol, clamp(sun * 0.5 + 0.5, 0.0, 1.0)) * lit;
            // The scoop is Nomex, and lets nothing through.
            float nomex = 1.0 - smoothstep(uMouthY - 0.06, uMouthY + 0.02, vP.y);
            col = mix(col, base * mix(uGround, uSky, 0.35) * 0.55, nomex);
            // the burner: strong at the mouth, dying up the cone
            float reach = (1.0 - smoothstep(0.0, 0.85, up));
            col += base * uFlame * uBurn * (0.12 + 1.05 * reach * reach);
          }
          gl_FragColor = vec4(col, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`
    });
  }

  /* A burner's flame is a jet: fast at the nozzle and held narrow by its own
   * speed, then it slows, widens and breaks up. Drawn as a body of
   * revolution whose brightness is what you would see through a volume -
   * most where you look through the most of it - with noise running up it
   * faster than the eye can follow, and additive, so it is light and not
   * paint. Blue where the propane has only just lit, at the foot. */
  function flameMaterial() {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 }, uPower: { value: 0 }, uSeed: { value: 0 }, uCore: { value: 0 }
      },
      vertexShader: `
        uniform float uTime, uPower, uSeed;
        varying vec2 vUv; varying vec3 vN; varying vec3 vV; varying vec3 vO;
        ${NOISE_GLSL}
        void main() {
          vUv = uv;
          vec3 p = position;
          float h = uv.y;
          float t = uTime + uSeed;
          // it whips more the further it is from the nozzle
          float w = h * h;
          p.x += w * (0.16 * sin(t * 11.0 + h * 6.0) + 0.09 * sin(t * 23.0 + h * 15.0));
          p.z += w * (0.16 * cos(t * 9.3 + h * 5.0) + 0.09 * cos(t * 19.0 + h * 13.0));
          float lump = fbm(vec3(p.xz * 3.0, h * 5.0 - t * 5.0));
          p.xz *= 1.0 + (lump - 0.5) * 0.7 * h;
          p.y *= mix(0.22, 1.0, uPower) * (0.94 + 0.08 * sin(t * 31.0));
          vO = p;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vN = normalize(normalMatrix * normal);
          vV = normalize(-mv.xyz);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform float uTime, uPower, uSeed, uCore;
        varying vec2 vUv; varying vec3 vN; varying vec3 vV; varying vec3 vO;
        ${NOISE_GLSL}
        void main() {
          float h = vUv.y;
          float t = uTime + uSeed;
          float a = vUv.x * 6.2831853;
          float n = fbm(vec3(cos(a) * 1.6, sin(a) * 1.6, h * 5.0 - t * 7.0));
          float n2 = fbm(vec3(cos(a) * 4.0 + 5.0, sin(a) * 4.0, h * 12.0 - t * 13.0));
          float thick = pow(abs(dot(normalize(vN), normalize(vV))), mix(1.3, 2.2, uCore));
          float foot = smoothstep(0.0, 0.035, h);
          float tip = 1.0 - smoothstep(0.42, 1.0, h + (n - 0.5) * 0.75);
          float body = foot * tip * (0.55 + 0.75 * n) * (0.75 + 0.5 * n2);
          float I = thick * body * uPower;
          vec3 hot = vec3(1.0, 0.93, 0.74), mid = vec3(1.0, 0.60, 0.16), cool = vec3(0.92, 0.26, 0.05);
          vec3 col = mix(mid, hot, clamp(thick * 1.3 - h * 0.9, 0.0, 1.0));
          col = mix(col, cool, smoothstep(0.45, 1.0, h + (0.5 - n) * 0.5));
          col = mix(col, hot, uCore * 0.6);
          // the foot burns blue
          col = mix(vec3(0.25, 0.48, 1.0), col, smoothstep(0.0, 0.11, h));
          gl_FragColor = vec4(col * I * 0.55, I * 0.4);
        }`
    });
  }

  /** A soft round glow, for the light a flame throws into the air round it. */
  function glowTexture() {
    const S = 128;
    const c = makeCanvas(S, S);
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.25, 'rgba(255,255,255,.55)');
    g.addColorStop(0.6, 'rgba(255,255,255,.14)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, S, S);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }

  /* ---------- the rig ---------- */

  function makeRig(canvas, opts = {}) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, premultipliedAlpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    const shadows = opts.shadows !== false;
    renderer.shadowMap.enabled = shadows;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const aniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(LENS.VFOV, 1, 0.04, 120);
    camera.rotation.order = 'YXZ';

    /* The sky the metal reflects. A dawn in three bands - the blue overhead,
     * the warm line of the horizon, the dark ground - blurred into an
     * environment map once. Without it stainless steel is grey paint. */
    {
      const env = new THREE.Scene();
      const sg = new THREE.SphereGeometry(10, 32, 16);
      const cols = [];
      const top = new THREE.Color(0x2c4a82), hor = new THREE.Color(0xffe0b4), low = new THREE.Color(0x5e5842);
      const sunward = new THREE.Color(0xffe0b0);
      const tmp = new THREE.Color();
      const p = sg.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const y = p.getY(i) / 10, x = p.getX(i) / 10;
        if (y >= 0) tmp.copy(hor).lerp(top, Math.pow(y, 0.55));
        else tmp.copy(hor).lerp(low, Math.min(1, -y * 4));   // the ground a plate looks down at is a sunlit town
        // the east is brighter: the sun is just up, on +x of this little sky
        tmp.lerp(sunward, Math.max(0, x) * Math.max(0, 1 - Math.abs(y - 0.15) * 2.4) * 0.8);
        cols.push(tmp.r, tmp.g, tmp.b);
      }
      sg.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      env.add(new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
      const pm = new THREE.PMREMGenerator(renderer);
      scene.environment = pm.fromScene(env, 0.04).texture;
      pm.dispose();
      sg.dispose();
    }

    const hemi = new THREE.HemisphereLight(0xa9c0e6, 0x6b5d49, 1.0);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffd3a1, 4.6);
    sun.castShadow = shadows;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = -1.7; sun.shadow.camera.right = 1.7;
    sun.shadow.camera.top = 2.2; sun.shadow.camera.bottom = -1.4;
    sun.shadow.camera.near = 1; sun.shadow.camera.far = 14;
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.012;
    sun.shadow.radius = 3;
    scene.add(sun, sun.target);
    sun.target.position.set(0, 1.5, 0);

    // The burner's light, on everything under it.
    const fire = new THREE.PointLight(0xffb060, 0, 9, 1.7);
    fire.position.set(0, G.COIL_TOP + 0.55, 0);

    const rig = new THREE.Group();
    scene.add(rig);
    /* Everything over your head - the frame, the burners and their fire, the
     * envelope - is one group, because on a tall screen it is moved: see
     * TALL, and placeUpper. */
    const upper = new THREE.Group();
    rig.add(upper);
    upper.add(fire);

    /* ----- materials ----- */
    const T = {};
    {
      const w = wicker();
      T.wicker = tex(w.map, { srgb: true, aniso });
      T.wickerN = tex(w.normal, { aniso });
      const l = leather(51, [156, 96, 54], true);
      T.leather = tex(l.map, { srgb: true, aniso });
      T.leatherN = tex(l.normal, { aniso });
      const l2 = leather(57, [104, 50, 38], false);
      T.suede = tex(l2.map, { srgb: true, aniso });
      T.suedeN = tex(l2.normal, { aniso });
      const s = sacking();
      T.sack = tex(s.map, { srgb: true, aniso, rx: 2 });
      T.sackN = tex(s.normal, { aniso, rx: 2 });
      const rp = rope([196, 170, 124]);
      T.rope = tex(rp.map, { srgb: true, aniso });
      T.ropeN = tex(rp.normal, { aniso });
      T.perf = tex(perforated(), { srgb: true, aniso, rx: 16, ry: 3 });
      const rr = rope([200, 44, 36]);
      T.red = tex(rr.map, { srgb: true, aniso });
    }
    const M = {
      wicker: new THREE.MeshStandardMaterial({
        map: T.wicker, normalMap: T.wickerN, normalScale: new THREE.Vector2(1.5, 1.5),
        roughness: 0.62, metalness: 0, envMapIntensity: 0.5
      }),
      rim: new THREE.MeshStandardMaterial({
        map: T.leather, normalMap: T.leatherN, normalScale: new THREE.Vector2(0.9, 0.9),
        roughness: 0.58, metalness: 0, envMapIntensity: 0.6
      }),
      sleeve: new THREE.MeshStandardMaterial({
        map: T.suede, normalMap: T.suedeN, normalScale: new THREE.Vector2(0.7, 0.7),
        roughness: 0.82, metalness: 0, envMapIntensity: 0.35
      }),
      steel: new THREE.MeshStandardMaterial({ color: 0xc9ccd0, roughness: 0.28, metalness: 1, envMapIntensity: 1.25 }),
      heat: new THREE.MeshStandardMaterial({ color: 0xb59a78, roughness: 0.36, metalness: 1, envMapIntensity: 1.2 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x1c1d20, roughness: 0.5, metalness: 0.6, envMapIntensity: 0.8 }),
      rubber: new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.7, metalness: 0, envMapIntensity: 0.4 }),
      red: new THREE.MeshStandardMaterial({ color: 0xc8281e, roughness: 0.45, metalness: 0.1, envMapIntensity: 0.6 }),
      brass: new THREE.MeshStandardMaterial({ color: 0xc9a44a, roughness: 0.3, metalness: 1, envMapIntensity: 1.1 }),
      wire: new THREE.MeshStandardMaterial({ color: 0x8d9299, roughness: 0.4, metalness: 1, envMapIntensity: 1.0 }),
      sack: new THREE.MeshStandardMaterial({
        map: T.sack, normalMap: T.sackN, normalScale: new THREE.Vector2(1.1, 1.1),
        roughness: 0.95, metalness: 0, envMapIntensity: 0.3
      }),
      can: new THREE.MeshStandardMaterial({
        map: T.perf, color: 0xd0d2d6, roughness: 0.32, metalness: 0.95, envMapIntensity: 1.1
      }),
      rope: new THREE.MeshStandardMaterial({
        map: T.rope, normalMap: T.ropeN, roughness: 0.9, metalness: 0, envMapIntensity: 0.3
      }),
      redRope: new THREE.MeshStandardMaterial({ map: T.red, normalMap: T.ropeN, roughness: 0.85, metalness: 0, envMapIntensity: 0.3 }),
      thong: new THREE.MeshStandardMaterial({ color: 0xb99a6a, roughness: 0.75, metalness: 0, envMapIntensity: 0.3 }),
      floor: new THREE.MeshStandardMaterial({ color: 0x2c2620, roughness: 0.9, metalness: 0 })
    };
    const cast = (m, c = true, r = true) => { m.castShadow = c; m.receiveShadow = r; return m; };

    /* ----- the basket ----- */
    const TILE = 0.46;                          // metres of wall one wicker tile covers
    const outer = loop(G.W, G.D, G.CORNER);
    const inner = loop(G.W - 2 * G.WALL, G.D - 2 * G.WALL, G.CORNER - G.WALL);
    rig.add(cast(new THREE.Mesh(wallGeo(outer, 0, G.H, TILE, false), M.wicker)));
    rig.add(cast(new THREE.Mesh(wallGeo(inner, 0.02, G.H, TILE, true), M.wicker)));
    {
      const f = new THREE.Mesh(new THREE.PlaneGeometry(G.W - 0.06, G.D - 0.06), M.floor);
      f.rotation.x = -Math.PI / 2;
      f.position.y = 0.025;
      f.receiveShadow = true;
      rig.add(f);
    }
    // The rim: a padded roll of leather laced over the top of the weave.
    const mid = loop(G.W - G.WALL, G.D - G.WALL, G.CORNER - G.WALL / 2, 12, 12);
    {
      const g = new THREE.TubeGeometry(loopCurve(mid, G.H + 0.012), 220, G.RIM, 14, true);
      T.leather.repeat.set(Math.round(mid.total / 0.22), 1);
      rig.add(cast(new THREE.Mesh(g, M.rim)));
      // The roll is laced on with a rawhide thong, whipped round it all the
      // way along: a turn every few centimetres, each a little aslant.
      {
        const curve = loopCurve(mid, G.H + 0.012);
        const n = Math.round(mid.total / 0.085);
        const lace = new THREE.InstancedMesh(new THREE.TorusGeometry(G.RIM + 0.0015, 0.0034, 6, 22), M.thong, n);
        const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), q2 = new THREE.Quaternion();
        const z = new THREE.Vector3(0, 0, 1), up = new THREE.Vector3(0, 1, 0), one = new THREE.Vector3(1, 1, 1);
        for (let i = 0; i < n; i++) {
          const t = (i + 0.5) / n;
          const pt = curve.getPointAt(t), tan = curve.getTangentAt(t);
          q.setFromUnitVectors(z, tan);
          q2.setFromAxisAngle(up, 0.38);
          q.multiply(q2);
          m4.compose(pt, q, one);
          lace.setMatrixAt(i, m4);
        }
        lace.castShadow = true;
        rig.add(lace);
      }
      // Under the roll, inside, the hide comes a hand's width down the weave.
      rig.add(cast(new THREE.Mesh(wallGeo(loop(G.W - 2 * G.WALL - 0.006, G.D - 2 * G.WALL - 0.006, G.CORNER - G.WALL), G.H - 0.085, G.H, 0.5, true, 1), M.sleeve)));
      // A scuff band of hide round the foot of the basket, where it lands.
      const foot = new THREE.Mesh(wallGeo(loop(G.W + 0.012, G.D + 0.012, G.CORNER + 0.006), 0, 0.16, 0.5, false, 1), M.sleeve);
      rig.add(cast(foot));
    }
    // Rope handles on the inside of the far wall, for the landing.
    {
      const handle = (x) => {
        const zin = -G.D / 2 + G.WALL + 0.004;
        const c = new THREE.CatmullRomCurve3([
          new THREE.Vector3(x - 0.13, G.H - 0.13, zin), new THREE.Vector3(x - 0.10, G.H - 0.19, zin + 0.035),
          new THREE.Vector3(x, G.H - 0.225, zin + 0.05), new THREE.Vector3(x + 0.10, G.H - 0.19, zin + 0.035),
          new THREE.Vector3(x + 0.13, G.H - 0.13, zin)
        ]);
        const g = new THREE.TubeGeometry(c, 24, 0.011, 8, false);
        const m = new THREE.Mesh(g, M.rope);
        return cast(m);
      };
      T.rope.repeat.set(10, 1);
      rig.add(handle(-0.36), handle(0.36));
    }

    /* ----- uprights and the burner frame ----- */
    const fx = G.FRAME_W / 2, fz = G.FRAME_D / 2;
    const bx = G.W / 2 - 0.085, bz = G.D / 2 - 0.085;
    const feet = [[-bx, -bz], [bx, -bz], [-bx, bz], [bx, bz]];
    const tops = [[-fx, -fz], [fx, -fz], [-fx, fz], [fx, fz]];
    T.suede.repeat.set(1, 5);
    // Each upright is built standing straight up from its foot, a nominal
    // length long, and then laid along the line from its socket on the rim to
    // its corner of the frame; when the frame moves it is laid again.
    const poles = [];
    const yUp = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < 4; i++) {
      const foot = new THREE.Vector3(feet[i][0], G.H - 0.25, feet[i][1]);
      const top = new THREE.Vector3(tops[i][0], G.FRAME_Y, tops[i][1]);
      const L = foot.distanceTo(top);
      const at = (t) => new THREE.Vector3(0, L * t, 0);
      const g = new THREE.Group();
      // The pole is nylon; what you see is its padded cover, a steel end
      // where the cover stops short of the frame, and the leather socket it
      // stands in on the rim.
      g.add(cast(rod(at(0.10), at(0.93), 0.034, M.sleeve, 14, 8)));
      g.add(cast(rod(at(0.93), at(1), 0.019, M.steel, 10)));
      g.add(cast(rod(at(0.918), at(0.938), 0.037, M.dark, 14)));
      g.add(cast(rod(at(0.10), at(0.26), 0.042, M.rim, 14)));
      g.position.copy(foot);
      rig.add(g);
      poles.push({ g, foot, top, L });
    }
    const _v = new THREE.Vector3();
    function placeUpper(dy, dz) {
      upper.position.set(0, dy, dz);
      for (const p of poles) {
        _v.copy(p.top).add(upper.position).sub(p.foot);
        const len = _v.length();
        p.g.quaternion.setFromUnitVectors(yUp, _v.normalize());
        p.g.scale.set(1, len / p.L, 1);
      }
    }
    placeUpper(0, 0);
    const frameLoop = loop(G.FRAME_W, G.FRAME_D, 0.07, 6, 4);
    upper.add(cast(new THREE.Mesh(new THREE.TubeGeometry(loopCurve(frameLoop, G.FRAME_Y), 96, 0.019, 10, true), M.steel)));
    // Two cross-members carry the burners between them.
    for (const z of [-0.105, 0.105]) {
      upper.add(cast(rod(new THREE.Vector3(-fx, G.FRAME_Y, z), new THREE.Vector3(fx, G.FRAME_Y, z), 0.014, M.steel, 8)));
    }
    // A karabiner at each corner: this is where the envelope is hung from.
    const hang = [];
    for (let i = 0; i < 4; i++) {
      const k = new THREE.Mesh(new THREE.TorusGeometry(0.036, 0.0075, 8, 20), M.steel);
      k.scale.set(0.62, 1.25, 1);
      k.position.set(tops[i][0], G.FRAME_Y + 0.052, tops[i][1]);
      k.rotation.y = Math.atan2(-tops[i][1], tops[i][0]) + Math.PI / 2;
      upper.add(cast(k, true, false));
      hang.push(new THREE.Vector3(tops[i][0], G.FRAME_Y + 0.095, tops[i][1]));
    }

    /* ----- the burners ----- */
    const levers = [];
    const nozzles = [];
    for (const side of [-1, 1]) {
      const b = new THREE.Group();
      b.position.set(side * G.BURNER_X, G.FRAME_Y, 0);
      // Seen from where you stand, a burner is mostly its underside: the
      // plate it is bolted through, the valve block with the hose coming in
      // and a pressure gauge on its face, a fixed steel handle, and the red
      // blast lever you squeeze up against it.
      const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.106, 0.106, 0.012, 32), M.steel);
      plate.position.y = -0.004;
      b.add(cast(plate));
      const block = new THREE.Mesh(new THREE.BoxGeometry(0.082, 0.085, 0.10), M.dark);
      block.position.y = -0.052;
      b.add(cast(block));
      const union = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.017, 0.04, 6), M.brass);
      union.position.set(-side * 0.02, -0.112, -0.03);
      b.add(union);
      const gauge = new THREE.Mesh(new THREE.CylinderGeometry(0.023, 0.023, 0.018, 20), M.brass);
      gauge.rotation.x = Math.PI / 2;
      gauge.position.set(0, -0.05, 0.058);
      b.add(gauge);
      const face = new THREE.Mesh(new THREE.CircleGeometry(0.0195, 20), new THREE.MeshStandardMaterial({ color: 0xf2efe6, roughness: 0.4 }));
      face.position.set(0, -0.05, 0.0675);
      b.add(face);
      const needle = new THREE.Mesh(new THREE.BoxGeometry(0.002, 0.016, 0.001), M.red);
      needle.position.set(0.004, -0.045, 0.0682);
      needle.rotation.z = -0.7;
      b.add(needle);
      // the fixed handle: a steel hoop out towards you
      const hx = side * 0.052;
      const hoop = new THREE.CatmullRomCurve3([
        new THREE.Vector3(hx, -0.03, 0.03), new THREE.Vector3(hx, -0.075, 0.075), new THREE.Vector3(hx, -0.10, 0.145),
        new THREE.Vector3(hx, -0.085, 0.165)
      ]);
      b.add(cast(new THREE.Mesh(new THREE.TubeGeometry(hoop, 16, 0.0065, 8, false), M.steel), true, false));
      const lever = new THREE.Group();
      lever.position.set(hx, -0.10, 0.035);
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.013, 0.010, 0.115), M.red);
      arm.position.set(0, -0.026, 0.062);
      arm.rotation.x = 0.36;
      lever.add(cast(arm, true, false));
      const grip = new THREE.Mesh(new THREE.SphereGeometry(0.014, 12, 10), M.red);
      grip.position.set(0, -0.048, 0.122);
      lever.add(grip);
      b.add(lever);
      levers.push(lever);
      // the can: a perforated stainless shroud round the jets
      const can = new THREE.Mesh(new THREE.CylinderGeometry(0.098, 0.082, 0.17, 36, 1, true), M.can);
      can.position.y = 0.085;
      b.add(cast(can));
      const canIn = new THREE.Mesh(new THREE.CylinderGeometry(0.094, 0.079, 0.17, 28, 1, true),
        new THREE.MeshStandardMaterial({ color: 0x24201c, roughness: 0.7, metalness: 0.5, side: THREE.BackSide }));
      canIn.position.y = 0.085;
      b.add(canIn);
      const lip = new THREE.Mesh(new THREE.TorusGeometry(0.098, 0.006, 8, 32), M.steel);
      lip.rotation.x = Math.PI / 2;
      lip.position.y = 0.17;
      b.add(lip);
      // the coil: the propane goes round the flame before it is burned, to
      // boil it. Heat has coloured the steel.
      const TURNS = 6.5, CR = 0.092, C0 = 0.18, C1 = G.COIL_TOP - G.FRAME_Y - 0.02;
      class Helix extends THREE.Curve {
        getPoint(t, out = new THREE.Vector3()) {
          const a = t * TURNS * Math.PI * 2;
          const rr = CR * (1 + 0.10 * t);
          return out.set(Math.cos(a) * rr, lerp(C0, C1, t), Math.sin(a) * rr);
        }
      }
      b.add(cast(new THREE.Mesh(new THREE.TubeGeometry(new Helix(), 200, 0.0115, 8, false), M.heat)));
      for (let k = 0; k < 3; k++) {
        const a = k / 3 * Math.PI * 2 + 0.5;
        const p0 = new THREE.Vector3(Math.cos(a) * 0.108, 0.16, Math.sin(a) * 0.108);
        const p1 = new THREE.Vector3(Math.cos(a) * 0.115, C1 + 0.01, Math.sin(a) * 0.115);
        b.add(cast(rod(p0, p1, 0.005, M.steel, 6)));
      }
      const ring = new THREE.Mesh(new THREE.TorusGeometry(CR * 1.1, 0.008, 8, 32), M.heat);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = C1 + 0.012;
      b.add(ring);
      // the jet ring inside, and the pilot light beside it
      const jets = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 0.06, 16), M.dark);
      jets.position.y = 0.12;
      b.add(jets);
      upper.add(b);
      nozzles.push(new THREE.Vector3(side * G.BURNER_X, G.FRAME_Y + 0.15, 0));
    }
    // Fuel hoses: out of each valve block, tied along the far cross-member
    // and the frame to the far left corner, and down that upright to the tanks.
    {
      const zc = -0.105;
      const FY = G.FRAME_Y;
      const down = (x0, off) => {
        const pole = (t, dx) => new THREE.Vector3(lerp(-fx, -bx, t) + 0.048 + dx, lerp(FY, G.H, t), lerp(-fz, -bz, t) + 0.045);
        const c = new THREE.CatmullRomCurve3([
          new THREE.Vector3(x0, FY - 0.115, -0.03),
          new THREE.Vector3(x0 - 0.005, FY - 0.165, -0.05),
          new THREE.Vector3(x0 - 0.05, FY - 0.14, -0.09),
          new THREE.Vector3(x0 - 0.11, FY - 0.06 - off, zc - 0.022),
          new THREE.Vector3(x0 - 0.22, FY - 0.036 - off, zc - 0.022 - off),
          new THREE.Vector3(lerp(x0 - 0.22, -fx + 0.2, 0.5), FY - 0.04 - off, zc - 0.022 - off),
          new THREE.Vector3(-fx + 0.20, FY - 0.036 - off, zc - 0.022 - off),
          new THREE.Vector3(-fx + 0.10, FY - 0.04 - off, zc - 0.06),
          new THREE.Vector3(-fx + 0.055 + off, FY - 0.05, lerp(zc, -fz, 0.6)),
          new THREE.Vector3(-fx + 0.05 + off, FY - 0.09, -fz + 0.06),
          pole(0.12, off), pole(0.3, off), pole(0.5, off), pole(0.7, off), pole(0.9, off),
          new THREE.Vector3(-bx + 0.085 + off, G.H - 0.02, -bz + 0.10),
          new THREE.Vector3(-bx + 0.09, G.H - 0.3, -bz + 0.13),
          new THREE.Vector3(-bx + 0.05, 0.8, -bz + 0.10)
        ], false, 'centripetal');
        return cast(new THREE.Mesh(new THREE.TubeGeometry(c, 180, 0.009, 8, false), M.rubber));
      };
      upper.add(down(-G.BURNER_X, 0), down(G.BURNER_X, 0.022));
      // a few ties holding them to the upright
      for (const t of [0.3, 0.5, 0.7]) {
        const tie = new THREE.Mesh(new THREE.TorusGeometry(0.045, 0.004, 6, 16), M.dark);
        tie.position.set(lerp(-fx, -bx, t) + 0.02, lerp(G.FRAME_Y, G.H, t), lerp(-fz, -bz, t) + 0.02);
        tie.rotation.x = Math.PI / 2;
        upper.add(tie);
      }
      // The tanks themselves: two cylinders in padded jackets in the far corners.
      for (const sx of [-1, 1]) {
        const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.155, 0.155, 0.86, 24), M.sleeve);
        tank.position.set(sx * (G.W / 2 - G.WALL - 0.17), 0.46, -G.D / 2 + G.WALL + 0.17);
        rig.add(cast(tank));
        const collar = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.012, 8, 24), M.steel);
        collar.rotation.x = Math.PI / 2;
        collar.position.set(tank.position.x, 0.93, tank.position.z);
        rig.add(collar);
      }
    }

    /* ----- the flames ----- */
    const flames = [];
    const FLAME_LEN = 3.3;
    {
      const outline = (rMax, len) => {
        const pts = [];
        const N = 40;
        for (let i = 0; i <= N; i++) {
          const t = i / N;
          // narrow at the nozzle, widest a third of the way, a long taper
          const r = rMax * (0.20 + 0.80 * Math.pow(Math.sin(Math.PI * Math.pow(t, 0.62)), 0.85)) * (1 - 0.25 * t);
          pts.push(new THREE.Vector2(Math.max(0.004, r * (t > 0.98 ? 0.3 : 1)), t * len));
        }
        return pts;
      };
      const gOut = new THREE.LatheGeometry(outline(0.27, FLAME_LEN), 28);
      nozzles.forEach((p, i) => {
        const mo = flameMaterial();
        mo.uniforms.uSeed.value = i * 3.7;
        const o = new THREE.Mesh(gOut, mo);
        o.position.copy(p);
        o.frustumCulled = false;
        o.renderOrder = 5;
        upper.add(o);
        flames.push(mo);
      });
      // The body of the fire. The shell above is what a flame looks like
      // from the side; from under it, which is where you are, a shell is
      // seen edge-on and is nearly nothing. So the flame itself is puffs of
      // burning gas rising up the jet, each facing the eye, and from below
      // they pile up into the white heart a burner has when you look up it.
      const PER = 34, N = PER * nozzles.length;
      const quad = new THREE.PlaneGeometry(1, 1);
      const g = new THREE.InstancedBufferGeometry();
      g.index = quad.index;
      g.setAttribute('position', quad.attributes.position);
      g.setAttribute('uv', quad.attributes.uv);
      const seed = new Float32Array(N * 4), base = new Float32Array(N * 3);
      const r = rng(88);
      for (let i = 0; i < N; i++) {
        const nz = nozzles[Math.floor(i / PER)];
        seed.set([(i % PER) / PER + r() * 0.03, r(), r(), r()], i * 4);
        base.set([nz.x, nz.y + 0.04, nz.z], i * 3);
      }
      g.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seed, 4));
      g.setAttribute('aBase', new THREE.InstancedBufferAttribute(base, 3));
      g.instanceCount = N;
      const pm = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        uniforms: { uTime: { value: 0 }, uPower: { value: 0 }, uLen: { value: FLAME_LEN } },
        vertexShader: `
          attribute vec4 aSeed; attribute vec3 aBase;
          uniform float uTime, uPower, uLen;
          varying vec2 vUv; varying float vH; varying float vS;
          void main() {
            float h = fract(aSeed.x + uTime * (1.5 + 0.9 * aSeed.w));
            float len = uLen * mix(0.16, 1.0, uPower);
            vec3 c = aBase + vec3(0.0, h * len, 0.0);
            float w = h * h, t = uTime;
            c.x += w * (0.20 * sin(t * 8.0 + aSeed.y * 31.0) + 0.10 * sin(t * 19.0 + aSeed.y * 77.0)) + (aSeed.z - 0.5) * 0.12 * h;
            c.z += w * (0.20 * cos(t * 7.1 + aSeed.y * 43.0) + 0.10 * cos(t * 17.0 + aSeed.y * 59.0)) + (fract(aSeed.z * 7.0) - 0.5) * 0.12 * h;
            float size = mix(0.12, 0.62, pow(sin(3.14159 * pow(h, 0.5)), 0.9)) * (0.75 + 0.5 * aSeed.z) * mix(0.45, 1.0, uPower);
            vec4 mv = modelViewMatrix * vec4(c, 1.0);
            float a = aSeed.y * 6.2831 + t * (aSeed.w - 0.5) * 3.0;
            vec2 q = vec2(position.x * cos(a) - position.y * sin(a), position.x * sin(a) + position.y * cos(a));
            mv.xy += q * size;
            gl_Position = projectionMatrix * mv;
            vUv = uv; vH = h; vS = aSeed.y;
          }`,
        fragmentShader: `
          uniform float uTime, uPower;
          varying vec2 vUv; varying float vH; varying float vS;
          ${NOISE_GLSL}
          void main() {
            vec2 p = vUv - 0.5;
            float r = length(p) * 2.0;
            float n = vnoise(vec3(p * 6.0 + vS * 10.0, uTime * 4.0 + vS * 5.0));
            float fall = smoothstep(1.0, 0.1, r + (n - 0.5) * 0.55);
            float life = smoothstep(0.0, 0.04, vH) * (1.0 - smoothstep(0.5, 1.0, vH));
            float I = fall * fall * life * uPower;
            vec3 hot = vec3(1.0, 0.95, 0.82), mid = vec3(1.0, 0.60, 0.16), cool = vec3(0.82, 0.20, 0.04);
            vec3 col = mix(hot, mid, smoothstep(0.04, 0.5, vH + r * 0.3));
            col = mix(col, cool, smoothstep(0.5, 1.0, vH + r * 0.2));
            col = mix(vec3(0.30, 0.52, 1.0), col, smoothstep(0.0, 0.06, vH));
            gl_FragColor = vec4(col * I * 0.6, I * 0.45);
          }`
      });
      const puffs = new THREE.Mesh(g, pm);
      puffs.frustumCulled = false;
      puffs.renderOrder = 7;
      upper.add(puffs);
      flames.push(pm);
    }
    // The pilot lights never go out: two small blue flames, always.
    const pilots = [];
    {
      const g = new THREE.ConeGeometry(0.016, 0.085, 10, 1, true);
      g.translate(0, 0.0425, 0);
      for (const p of nozzles) {
        const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
          color: 0x4f8cff, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false
        }));
        m.position.set(p.x + 0.03, p.y + 0.03, 0.03);
        upper.add(m);
        pilots.push(m);
      }
    }
    // And the air round a flame glows.
    const glowTex = glowTexture();
    const glows = [];
    for (const [y, s] of [[0.5, 1.5], [1.5, 2.3], [2.5, 1.9]]) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTex, color: 0xff8a30, transparent: true, opacity: 0, blending: THREE.AdditiveBlending,
        depthWrite: false, depthTest: true
      }));
      sp.position.set(0, G.COIL_TOP + y, 0);
      sp.scale.set(s, s, 1);
      sp.renderOrder = 4;
      upper.add(sp);
      glows.push(sp);
    }

    /* ----- the envelope ----- */
    const profile = envelopeProfile();
    const fabricTex = tex(fabric(profile.vMouth), { srgb: true, aniso });
    fabricTex.wrapT = THREE.ClampToEdgeWrapping;
    const envMat = envelopeMaterial(fabricTex);
    const envelope = new THREE.Mesh(envelopeGeo(profile), envMat);
    envelope.frustumCulled = false;
    const sky = new THREE.Group();              // everything that hangs above the frame and lags the basket
    sky.add(envelope);
    upper.add(sky);

    // The flying wires: from the four karabiners to the load tapes at the mouth.
    {
      for (let i = 0; i < G.GORES; i++) {
        const a = (i + 0.5) / G.GORES * Math.PI * 2;
        const top = new THREE.Vector3(Math.cos(a) * G.MOUTH_R * 0.995, G.MOUTH_Y, Math.sin(a) * G.MOUTH_R * 0.995);
        // each wire goes to the corner on its own side of the frame
        let best = 0, bd = Infinity;
        for (let k = 0; k < 4; k++) {
          const d = Math.hypot(hang[k].x - top.x, hang[k].z - top.z);
          if (d < bd) { bd = d; best = k; }
        }
        sky.add(rod(hang[best], top, 0.0032, M.wire, 5));
      }
    }

    // The crown. The top of the envelope is open, and a round panel - the
    // parachute - is held up against the hole from inside by the pressure.
    // The red line pulls it down, and the hot air goes out round it.
    const crownY = profile.pts[profile.pts.length - 1].y;
    const crownR = profile.pts[profile.pts.length - 1].x;
    const chute = new THREE.Group();
    {
      const cg = new THREE.CircleGeometry(crownR * 1.32, 48, 0, Math.PI * 2);
      const cc = makeCanvas(256, 256);
      const cx = cc.getContext('2d');
      cx.fillStyle = '#f4ead2';
      cx.fillRect(0, 0, 256, 256);
      cx.translate(128, 128);
      for (let i = 0; i < 24; i++) {
        cx.rotate(Math.PI * 2 / 24);
        cx.fillStyle = i % 2 ? 'rgba(214,62,38,.9)' : 'rgba(253,205,1,.9)';
        cx.beginPath(); cx.moveTo(0, 0); cx.arc(0, 0, 128, 0, Math.PI * 2 / 24); cx.closePath(); cx.fill();
        cx.fillStyle = 'rgba(30,20,14,.7)';
        cx.fillRect(0, -1.5, 128, 3);
      }
      const ct = new THREE.CanvasTexture(cc);
      ct.colorSpace = THREE.SRGBColorSpace;
      const disc = new THREE.Mesh(cg, new THREE.MeshBasicMaterial({ map: ct, side: THREE.DoubleSide }));
      disc.rotation.x = Math.PI / 2;
      chute.add(disc);
      chute.userData.disc = disc;
      chute.position.y = crownY - 0.08;
      sky.add(chute);
      // The sky through the hole, when the parachute is down: a bright disc
      // above it. It is the only place the canvas shows the morning itself.
      const hole = new THREE.Mesh(new THREE.CircleGeometry(crownR * 1.02, 40),
        new THREE.MeshBasicMaterial({ color: 0xdfeaff, side: THREE.DoubleSide }));
      hole.rotation.x = Math.PI / 2;
      hole.position.y = crownY + 0.02;
      sky.add(hole);
    }
    // The red line, crown to basket. It comes down the middle of the
    // envelope, over a pulley on the frame, and ends in a handle in reach.
    let ventTail = null;
    {
      const top = new THREE.Vector3(0.0, crownY - 0.2, 0.0);
      const at = new THREE.Vector3(G.FRAME_W / 2 - 0.10, G.FRAME_Y + 0.02, -G.FRAME_D / 2 + 0.06);
      T.red.repeat.set(1, 1);
      const long = rod(at, top, 0.006, M.redRope, 6);
      sky.add(long);
      const tail = new THREE.Group();
      tail.position.copy(at);
      const t1 = rod(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -0.24, 0), 0.006, M.redRope, 6);
      const toggle = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 0.08, 10), M.red);
      toggle.rotation.z = Math.PI / 2;
      toggle.position.y = -0.25;
      tail.add(t1, cast(toggle, true, false));
      upper.add(tail);
      ventTail = tail;
    }

    /* ----- sandbags ----- */
    const bags = [];
    {
      // A sack tied at the neck: full and round at the bottom, cinched, and
      // the mouth of it standing up in a tuft above the tie.
      const prof = [
        [0.0, -0.335], [0.05, -0.333], [0.095, -0.315], [0.122, -0.275], [0.132, -0.22], [0.128, -0.16],
        [0.112, -0.10], [0.085, -0.055], [0.05, -0.025], [0.028, -0.008], [0.024, 0.0], [0.03, 0.01],
        [0.05, 0.035], [0.058, 0.05]
      ].map(([r, y]) => new THREE.Vector2(r, y));
      const bg = new THREE.LatheGeometry(prof, 20);
      // a sack is never round: knock it about a little
      {
        const p = bg.attributes.position;
        const n = tileNoise(71, 8, 8);
        for (let i = 0; i < p.count; i++) {
          const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
          const a = Math.atan2(z, x);
          const k = 1 + 0.16 * (n((a / Math.PI + 1) * 4, (y + 0.34) * 14) - 0.5) * (y < -0.02 ? 1 : 0.3);
          p.setXYZ(i, x * k, y, z * k);
        }
        bg.computeVertexNormals();
      }
      const tie = new THREE.TorusGeometry(0.027, 0.007, 6, 14);
      const place = (x, y, z, len, face, scale = 1) => {
        const pivot = new THREE.Group();
        pivot.position.set(x, y, z);
        const line = rod(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -len, 0), 0.0055, M.rope, 6);
        const body = new THREE.Mesh(bg, M.sack);
        body.position.y = -len;
        body.rotation.y = face;
        body.scale.setScalar(scale);
        const knot = new THREE.Mesh(tie, M.rope);
        knot.rotation.x = Math.PI / 2;
        knot.position.y = -len;
        pivot.add(cast(line, true, false), cast(body), knot);
        rig.add(pivot);
        bags.push({
          pivot, len: len + 0.2,
          ax: 0, vx: 0, az: 0, vz: 0,
          w: Math.sqrt(9.81 / (len + 0.2)) * (0.92 + 0.16 * Math.random()),
          ph: Math.random() * 6.28
        });
      };
      const zf = -G.D / 2 - 0.135;              // just outside the far wall
      const xs = G.W / 2 + 0.135;
      // Along the far wall, hung from the rim, and the pairs at the far
      // corners hung higher, from the uprights, where they show above it.
      place(-0.30, G.H + 0.07, zf, 0.20, Math.PI / 2 + 0.2);
      place(0.02, G.H + 0.07, zf, 0.24, Math.PI / 2 - 0.3, 0.94);
      place(0.34, G.H + 0.07, zf, 0.19, Math.PI / 2 + 0.5, 1.04);
      place(-bx + 0.02, G.H + 0.36, zf + 0.03, 0.17, Math.PI / 2 + 0.1, 1.02);
      place(bx - 0.03, G.H + 0.34, zf + 0.03, 0.15, Math.PI / 2 - 0.2, 0.98);
      // and down the two sides
      place(-xs, G.H + 0.07, -0.22, 0.22, 0.2);
      place(-xs, G.H + 0.07, 0.14, 0.19, -0.3, 0.95);
      place(xs, G.H + 0.07, -0.26, 0.20, Math.PI + 0.3, 1.03);
      place(xs, G.H + 0.07, 0.10, 0.23, Math.PI - 0.2);
      // each line is hitched round the rim (or the upright): a turn of rope
      const hitch = new THREE.TorusGeometry(G.RIM + 0.007, 0.0055, 6, 18);
      for (const b of bags) {
        const p = b.pivot.position;
        const h = new THREE.Mesh(hitch, M.rope);
        if (p.y > G.H + 0.2) {
          h.geometry = new THREE.TorusGeometry(0.047, 0.0055, 6, 16);
          h.rotation.x = Math.PI / 2;
          h.position.set(p.x, p.y - 0.01, p.z + 0.11);
          rig.add(rod(new THREE.Vector3(p.x, p.y - 0.01, p.z + 0.07), p, 0.0055, M.rope, 6));
        } else if (Math.abs(p.z - zf) < 0.01) {
          h.rotation.y = Math.PI / 2;
          h.position.set(p.x, G.H + 0.012, -G.D / 2 + G.WALL / 2);
          rig.add(rod(new THREE.Vector3(p.x, G.H + 0.06, -G.D / 2 - 0.02), p, 0.0055, M.rope, 6));
        } else {
          h.position.set(Math.sign(p.x) * (G.W / 2 - G.WALL / 2), G.H + 0.012, p.z);
          rig.add(rod(new THREE.Vector3(Math.sign(p.x) * (G.W / 2 + 0.02), G.H + 0.06, p.z), p, 0.0055, M.rope, 6));
        }
        rig.add(h);
      }
    }

    // A handling line, coiled and hung on the near right upright.
    {
      const coil = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.02, 8, 28), M.rope);
      coil.position.set(bx + 0.02, G.H + 0.02, bz - 0.25);
      coil.rotation.set(0.25, Math.PI / 2, 0);
      coil.scale.set(1, 1.5, 1);
      rig.add(cast(coil));
    }

    /* ----- motion ----- */
    let time = 0;
    let wide = true, fov0 = LENS.VFOV;
    const state = { burn: 0, flick: 1 };
    const head = { x: 0, y: 0, vx: 0, vy: 0 };   // the head's lag behind the basket, metres
    const lag = { roll: 0, pit: 0 };             // the envelope's lag behind the basket, degrees
    let lever = 0, pull = 0, glance = 0;
    const eye = new THREE.Vector3();
    const sunDir = new THREE.Vector3();

    function fitLens(w, h) {
      const aspect = w / h;
      const was = wide;
      wide = aspect >= 1;
      if (wide !== was) placeUpper(wide ? 0 : LENS.TALL[0], wide ? 0 : LENS.TALL[1]);
      const vForH = 2 * Math.atan(Math.tan(LENS.HFOV_MIN / 2 * RAD) / aspect) / RAD;
      fov0 = clamp(Math.max(LENS.VFOV, vForH), LENS.VFOV, LENS.VFOV_MAX);
      camera.fov = fov0;
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    }

    function resize(w, h, dpr) {
      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);
      fitLens(w, h);
    }

    /** One frame of everything that moves. `s` is the flight, from
     *  explore.js: burn 0..1 (the eased glow), burning and vent (the keys),
     *  roll, pit, rollV, pitV in degrees (the basket's swing), bearing in
     *  degrees, look as pitchTrim over its range, -1 down .. +1 up. */
    function update(s, dt) {
      time += dt;
      const burn = clamp(s.burn || 0, 0, 1);
      // The flame is not steady and neither is its light: a fast flutter on
      // a slower one, the way a jet of burning gas breathes.
      const flick = 0.80 + 0.13 * Math.sin(time * 37.0) * Math.sin(time * 11.3) + 0.07 * Math.sin(time * 71.0);
      state.burn = burn;
      state.flick = flick;

      for (const m of flames) {
        m.uniforms.uTime.value = time;
        m.uniforms.uPower.value = burn;
      }
      const lit = burn * flick;
      fire.intensity = 8.5 * lit;
      fire.position.y = G.COIL_TOP + 0.35 + 0.5 * burn;
      glows[0].material.opacity = 0.75 * lit;
      glows[1].material.opacity = 0.42 * lit * burn;
      glows[2].material.opacity = 0.24 * lit * burn;
      for (let i = 0; i < pilots.length; i++) {
        pilots[i].scale.y = 0.85 + 0.25 * Math.sin(time * 19 + i * 2.1);
        pilots[i].material.opacity = 0.8 * (1 - 0.7 * burn);
      }
      envMat.uniforms.uBurn.value = lit;
      envMat.uniforms.uTime.value = time;

      // The hand on the blast lever, and on the red line.
      lever += ((s.burning ? 1 : 0) - lever) * clamp(dt * 14, 0, 1);
      for (const l of levers) l.rotation.x = -0.30 * lever;
      pull += ((s.vent ? 1 : 0) - pull) * clamp(dt * 5, 0, 1);
      ventTail.position.y = G.FRAME_Y + 0.02 - 0.20 * pull;
      chute.position.y = crownY - 0.08 - 1.5 * pull;
      chute.userData.disc.scale.setScalar(1 - 0.42 * pull);

      // The sun is in the east and low. The rig faces `bearing`, so in the
      // rig's own frame the east is that far round to the right.
      const az = (90 - (s.bearing || 0)) * RAD;
      const el = 16 * RAD;
      sunDir.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el));
      sun.position.copy(sunDir).multiplyScalar(7).add(sun.target.position);
      envMat.uniforms.uSun.value.copy(sunDir);

      // What hangs, hangs from the true vertical, and the basket has swung
      // away from that by its roll and pitch. Each bag is a short pendulum of
      // its own, driven towards that vertical and never quite settling.
      const roll = s.roll || 0, pit = s.pit || 0;
      for (const b of bags) {
        const gust = 0.9 * Math.sin(time * 0.9 + b.ph) + 0.5 * Math.sin(time * 2.3 + b.ph * 1.7);
        const tz = -roll * 1.6 + gust, tx = pit * 1.6 + 0.6 * Math.sin(time * 1.3 + b.ph * 2.1);
        const k = b.w * b.w, c = 2 * 0.06 * b.w;
        b.vz += (k * (tz - b.az) - c * b.vz) * dt;
        b.az += b.vz * dt;
        b.vx += (k * (tx - b.ax) - c * b.vx) * dt;
        b.ax += b.vx * dt;
        b.pivot.rotation.z = b.az * RAD;
        b.pivot.rotation.x = b.ax * RAD;
      }
      ventTail.rotation.z = -roll * 1.4 * RAD + 0.02 * Math.sin(time * 1.1);
      ventTail.rotation.x = pit * 1.4 * RAD + 0.02 * Math.sin(time * 0.8);

      // The envelope is twenty metres of cloth on wires, and the basket
      // leads it: it follows a moment late.
      lag.roll += (roll - lag.roll) * clamp(dt / 0.9, 0, 1);
      lag.pit += (pit - lag.pit) * clamp(dt / 0.9, 0, 1);
      sky.rotation.z = (lag.roll - roll) * 0.9 * RAD;
      sky.rotation.x = -(lag.pit - pit) * 0.9 * RAD;

      // The head. It rides the basket, loosely: a spring to where it should
      // be, pushed by the basket's own swing, and a little of the burner's
      // shake while it is lit.
      const kx = 30, cx = 7;
      head.vx += (-kx * head.x - cx * head.vx - (s.rollV || 0) * 0.012) * dt;
      head.x += head.vx * dt;
      head.vy += (-kx * head.y - cx * head.vy + (s.pitV || 0) * 0.008) * dt;
      head.y += head.vy * dt;
      const shake = burn * 0.0016;

      // Where it looks. W leans out over the rim, S tips back.
      const look = clamp(s.look || 0, -1, 1);
      const dn = smooth(Math.max(0, -look)), upk = smooth(Math.max(0, look));
      // The glance. Nobody opens a burner without looking at it: the eyes go
      // up to the flame while it is lit and come back to the ground after.
      // Not while leaning out, and not on top of a head already tipped back.
      const gw = s.burning ? 1 : 0;
      glance += (gw - glance) * clamp(dt / (gw ? LENS.GLANCE_IN : LENS.GLANCE_OUT), 0, 1);
      const gl = smooth(glance) * (1 - dn) * (1 - upk);
      state.glance = gl;
      const e0 = wide ? LENS.EYE : LENS.EYE_TALL;
      eye.set(lerp(e0[0], LENS.LEAN[0], dn), lerp(e0[1], LENS.LEAN[1], dn), lerp(e0[2], LENS.LEAN[2], dn));
      eye.x += clamp(head.x, -0.05, 0.05) + shake * Math.sin(time * 61);
      eye.y += clamp(head.y, -0.04, 0.04) + 0.004 * Math.sin(time * 0.9) + shake * Math.sin(time * 53);
      const fov = lerp(fov0, Math.min(fov0, LENS.VFOV_UP * fov0 / LENS.VFOV), upk);
      if (Math.abs(fov - camera.fov) > 0.01) {
        camera.fov = fov;
        camera.updateProjectionMatrix();
      }
      camera.position.copy(eye);
      camera.rotation.set(
        (LENS.PITCH + LENS.UP * upk + (wide ? LENS.GLANCE : LENS.GLANCE_TALL) * gl - LENS.DOWN * dn + pit * LENS.PITCH_K) * RAD,
        (wide ? LENS.YAW : LENS.YAW * 0.3) * (1 - dn) * RAD,
        roll * LENS.ROLL_K * RAD
      );
    }

    function render() { renderer.render(scene, camera); }

    /** For the tests: draw, and say how opaque the picture is at each of
     *  these places, given as fractions of the canvas from its top left.
     *  Read in the same breath as the draw, because a WebGL canvas forgets
     *  its picture once the browser has shown it. */
    function alphaAt(pts) {
      renderer.render(scene, camera);
      const gl = renderer.getContext();
      const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
      const px = new Uint8Array(4);
      return pts.map(([x, y]) => {
        gl.readPixels(Math.floor(x * (w - 1)), Math.floor((1 - y) * (h - 1)), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        return px[3];
      });
    }

    return {
      renderer, scene, camera, rig, resize, update, render, alphaAt,
      /** For the tests and the tuning page. */
      state, LENS, G, sun, fire
    };
  }

  return { load, ready: () => !!THREE, makeRig, LENS, G };
})();
