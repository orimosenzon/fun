// A jointed karateka for the acrobatics, drawn in the manner of the game's own sprites:
// white gi, black belt, skin as single-pixel stripes, and a black line wherever one part
// of the body passes in front of another. A pose is a handful of joint angles; the poses
// themselves are keyed by eye from slow-motion footage of gymnasts (see tools/flip_frames.mjs).
//
// Angles are in degrees. A direction of 0 points down, 90 forward (the fighter faces right),
// 180 up. Limb angles are relative to the body, so a pose can be turned as a whole (R).
export const W = 72, H = 60, CX = 29, CY = 30;   // grid; the hip sits at (CX, CY) before the frame's own offset

const LEN = { torso: 10.4, neck: 4.2, upper: 6.6, fore: 6.6, thigh: 10.6, shin: 10.6, foot: 4.6 };   // measured on the game's own fighter
// the hero's head in the fighting stance, copied by eye from the screen: hair, face, a dark eye, the neck
const HEAD = ['.######', '######.', '####ss.', '##sso..', '##ssss.', '##ssss.', '.sssss.', '.sss...'];
const rad = (a) => a * Math.PI / 180;
const dir = (a) => [Math.sin(rad(a)), Math.cos(rad(a))];
const add = (p, d, l) => [p[0] + d[0] * l, p[1] + d[1] * l];

// pose: { R, T, head, arm: [S, E], arm2, leg: [H, K, F], leg2 }  (arm2/leg2 = the far side, default = near side)
//   S shoulder (0 = arm along the body, 180 = overhead), E elbow, H hip (90 = thigh straight forward),
//   K knee bend, F direction of the foot (default: square to the shin)
//   reach: the near leg drawn longer, as the game does with a leg kicking towards the opponent
//   wide: the body turned towards the camera
//   flip: the whole figure faces left (his back to the opponent); lookBack: the head turned the other way
export function joints(p) {
  const R = p.R ?? 0, T = (p.T ?? 180) + R, down = T - 180;
  const hip = [0, 0], neck = add(hip, dir(T), LEN.torso), head = add(neck, dir(T + (p.head ?? 0)), LEN.neck);
  const shoulder = add(hip, dir(T), LEN.torso - 2);
  const arm = ([S, E]) => { const ua = down + S, elbow = add(shoulder, dir(ua), LEN.upper), hand = add(elbow, dir(ua + E), LEN.fore); return { elbow, hand, fa: ua + E }; };
  const leg = ([Hp, K, F], k = 1) => { const th = down + Hp, knee = add(hip, dir(th), LEN.thigh * k), ankle = add(knee, dir(th - K), LEN.shin * k), toe = add(ankle, dir(F === undefined ? th - K + 90 : F + R), LEN.foot); return { knee, ankle, toe }; };
  return { hip, neck, head, shoulder, T, headA: T + (p.head ?? 0), near: { ...arm(p.arm), ...leg(p.leg, p.reach) }, far: { ...arm(p.arm2 ?? p.arm), ...leg(p.leg2 ?? p.leg) } };
}

export function draw(p, { dx = 0, dy = 0 } = {}) {
  const g = Array.from({ length: H }, () => Array(W).fill('.'));
  const sx = p.flip ? -1 : 1, faceLeft = !!p.flip !== !!p.lookBack;
  const j = joints(p), O = ([x, y]) => [CX + dx + sx * x, CY + dy + y];
  const opaque = (ch) => ch !== '.';
  // a tapered limb between two points; `line` draws the black edge where it covers what is already there
  function limb(a, b, w1, w2, ch, line = true, joint = 0) {   // joint: no edge this close to the limb's root
    const [x1, y1] = O(a), [x2, y2] = O(b), ex = x2 - x1, ey = y2 - y1, L = ex * ex + ey * ey || 1e-6;
    const inside = (c, r, grow) => { let u = ((c - x1) * ex + (r - y1) * ey) / L; u = Math.max(0, Math.min(1, u)); return Math.hypot(c - x1 - u * ex, r - y1 - u * ey) <= (w1 + (w2 - w1) * u) / 2 + grow; };
    const x0 = Math.floor(Math.min(x1, x2) - 9), xe = Math.ceil(Math.max(x1, x2) + 9), y0 = Math.floor(Math.min(y1, y2) - 9), ye = Math.ceil(Math.max(y1, y2) + 9);
    const cells = [];
    for (let r = Math.max(0, y0); r <= Math.min(H - 1, ye); r++) for (let c = Math.max(0, x0); c <= Math.min(W - 1, xe); c++) {
      if (inside(c, r, 0)) cells.push([r, c, ch]);
      else if (line && opaque(g[r][c]) && inside(c, r, 0.95) && Math.hypot(c - x1, r - y1) > joint) cells.push([r, c, 'o']);
    }
    for (const [r, c, v] of cells) g[r][c] = v;
  }
  const side = (s, line) => {
    limb(j.shoulder, s.elbow, 4.4, 3.9, '#', line, 4.5); limb(s.elbow, s.hand, 3.9, 3.3, '#', line, 3.2);
    limb(s.hand, add(s.hand, dir(s.fa), 1.8), 3.2, 3, 's', false);
  };
  const leg = (s, line) => {
    limb(j.hip, s.knee, 6.8, 5.6, '#', line, 7); limb(s.knee, s.ankle, 5.5, 4.3, '#', line, 4.4);
    limb(s.ankle, s.toe, 2.7, 2.2, 's', false);
  };
  const parts = {
    arm2: () => side(j.far, false), leg2: () => leg(j.far, false),
    torso: () => {
      const w = p.wide ?? 8.4;                                          // wider when the body is turned to the camera
      limb(add(j.hip, dir(j.T), -2.2), j.shoulder, w, w + 0.4, '#', p.torsoLine ?? false);
      limb(j.shoulder, j.neck, w - 1.6, 5, '#', false);
      const t = dir(j.T), n = [-t[1], t[0]], b = add(j.hip, t, 3.4);   // belt, across the body
      limb(add(b, n, -w / 2 - 0.6), add(b, n, w / 2 + 0.6), 1.5, 1.5, 'o', false);
      if (p.wide) limb(add(add(j.neck, t, -2.6), n, 1.2), add(add(j.neck, t, -5.2), n, 0.4), 2.2, 1.4, 's', false);   // the open collar
    },
    head: () => {
      const t = dir(j.headA), f = [-t[1], t[0]];         // f = the way the face looks
      const tilt = ((j.headA - 180) % 360 + 540) % 360 - 180;
      const [hx, hy] = O(j.head);
      if (Math.abs(tilt) <= 40) {            // upright enough: the hero's own head, as the game draws it
        const x0 = 2 * Math.round((hx - 3) / 2), y0 = Math.round(hy - 4);
        HEAD.forEach((row, i) => [...(faceLeft ? [...row].reverse().join('') : row)].forEach((ch, k) => { if (ch !== '.' && g[y0 + i]?.[x0 + k] !== undefined) g[y0 + i][x0 + k] = ch; }));
        return;
      }
      limb(add(j.neck, t, -0.5), add(j.neck, t, 2), 3.4, 3.4, 's', false);
      // turned over: the same idea by rule, white hair over the top and down the back, the face in skin
      for (let r = Math.floor(hy - 6); r <= hy + 6; r++) for (let c = Math.floor(hx - 6); c <= hx + 6; c++) {
        if (r < 0 || r >= H || c < 0 || c >= W) continue;
        const X = (c - hx) * sx, u = X * t[0] + (r - hy) * t[1], v = X * f[0] + (r - hy) * f[1];   // u: towards the crown, v: towards the face
        if ((u / 4.1) ** 2 + (v / 3.1) ** 2 > 1) continue;
        g[r][c] = v > -0.7 && u < 1.3 ? 's' : '#';
      }
    },
    leg: () => leg(j.near, true), arm: () => side(j.near, true),
  };
  for (const k of p.order ?? ['arm2', 'leg2', 'torso', 'head', 'leg', 'arm']) parts[k]();
  return finish(g);
}

// The Apple II shows a lone lit pixel, or a lone dark one between two lit ones, as a colour.
// White needs two lit pixels side by side and a black line two dark ones.
export function finish(g) {
  for (const row of g) {
    for (let c = 1; c < W - 1; c++) if (row[c] === 'o' && row[c - 1] === '#' && row[c + 1] === '#') {
      if (row[c + 2] === '#' && row[c + 3] === '#') row[c + 1] = 'o'; else if (row[c - 2] === '#' && row[c - 3] === '#') row[c - 1] = 'o'; else row[c] = '#';
    }
    for (let c = 1; c < W - 1; c++) if (row[c] === '#' && row[c - 1] !== '#' && row[c + 1] !== '#' && row[c - 1] !== 's' && row[c + 1] !== 's') {
      if (row[c + 1] === '.') row[c + 1] = '#'; else if (row[c - 1] === '.') row[c - 1] = '#';
    }
  }
  return g.map((row) => row.map((ch, c) => ch === 's' ? (c & 1 ? '#' : 'o') : ch).join(''));
}

// a text patch on a finished frame: [[row, col, 'text'], ...], blanks leave the pixel alone
export function patch(rows, edits = []) {
  const g = rows.map((r) => [...r]);
  for (const [r, c, s] of edits) [...s].forEach((ch, i) => { if (ch !== ' ' && g[r]?.[c + i] !== undefined) g[r][c + i] = ch; });
  return g.map((r) => r.join(''));
}
