// ציור: מפה מלמעלה, מבט המצלמה של הטלפון (raycasting), ופרצוף על המסך.
import { rayAll, rayNearest } from './world.js';
import { FOV, RL_RAYS } from './modes.js';
import { BODY_R } from './robot.js';

let theme = null;
export function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim();
  theme = {
    floor: v('--floor'), wall: v('--wall'), ink: v('--ink'), muted: v('--muted'), accent: v('--accent'),
    unknown: v('--map-unknown'), bad: v('--bad'), warn: v('--warn'), good: v('--good'), panel: v('--panel'),
    dark: matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light' || document.documentElement.dataset.theme === 'dark',
  };
  return theme;
}

const rgb = (c, k = 1, a = 1) => `rgba(${c[0] * k | 0},${c[1] * k | 0},${c[2] * k | 0},${a})`;

export class MapView {
  constructor(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.grid = document.createElement('canvas');
  }
  fit(world) {
    const cssW = this.cv.clientWidth || 800;
    const aspect = world.H / world.W;
    const cssH = Math.round(cssW * aspect);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (this.cv.width !== Math.round(cssW * dpr) || this.cv.height !== Math.round(cssH * dpr)) {
      this.cv.width = Math.round(cssW * dpr); this.cv.height = Math.round(cssH * dpr);
      this.cv.style.height = cssH + 'px';
    }
    this.s = this.cv.width / world.W;
    this.world = world;
  }
  toWorld(px, py) {
    const r = this.cv.getBoundingClientRect();
    const k = this.cv.width / r.width;
    return { x: this.world.x0 + (px - r.left) * k / this.s, y: this.world.y0 + (py - r.top) * k / this.s };
  }
  draw(st) {
    const { ctx } = this; const w = this.world; const s = this.s; const T = theme || readTheme();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = T.floor; ctx.fillRect(0, 0, this.cv.width, this.cv.height);
    ctx.setTransform(s, 0, 0, s, -w.x0 * s, -w.y0 * s);
    const px = 1 / s;

    if (w.kind === 'arena') {
      ctx.fillStyle = T.dark ? '#1a1d22' : '#d9d4ca'; ctx.fillRect(w.x0, w.y0, w.W, w.H);
      ctx.beginPath(); ctx.arc(0, 0, w.R, 0, Math.PI * 2);
      ctx.fillStyle = T.dark ? '#6b4a2e' : '#c89a6a'; ctx.fill();
      ctx.lineWidth = 6 * px; ctx.strokeStyle = T.dark ? '#d8c3a5' : '#7a5232'; ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, w.R - 0.3, 0, Math.PI * 2);
      ctx.setLineDash([0.06, 0.06]); ctx.lineWidth = 2 * px; ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.stroke(); ctx.setLineDash([]);
    } else {
      for (const r of w.rooms) {
        ctx.fillStyle = rgb(r.tint, T.dark ? 0.35 : 1, T.dark ? 0.35 : 0.55);
        ctx.fillRect(r.x, r.y, r.w, r.h);
      }
      for (const r of w.rooms) this.label(r.name, r.x + r.w / 2, r.y + r.h / 2, 13, T.muted);
      // עגינה
      ctx.fillStyle = T.good; ctx.globalAlpha = 0.5;
      ctx.fillRect(w.dock.x - 0.2, w.dock.y - 0.15, 0.3, 0.3); ctx.globalAlpha = 1;
      for (const b of w.boxes) {
        ctx.fillStyle = rgb(b.color, T.dark ? 0.75 : 1, 0.9);
        ctx.fillRect(b.x, b.y, b.w, b.d);
      }
      const known = st.showMap && st.grid;
      if (!known) for (const b of w.boxes) if (b.w > 0.55) this.label(b.label, b.x + b.w / 2, b.y + b.d / 2, 11, '#fff');
      if (known) this.drawGrid(st.grid, T);
      ctx.strokeStyle = T.wall; ctx.lineWidth = 6 * px; ctx.lineCap = 'round';
      ctx.beginPath();
      for (const sg of w.segs) if (sg.kind === 'wall') { ctx.moveTo(sg.x1, sg.y1); ctx.lineTo(sg.x2, sg.y2); }
      ctx.stroke();
    }

    if (st.path && st.path.length) {
      ctx.strokeStyle = T.accent; ctx.lineWidth = 2 * px; ctx.setLineDash([6 * px, 5 * px]);
      ctx.beginPath(); ctx.moveTo(st.robots[0].x, st.robots[0].y);
      for (const p of st.path) ctx.lineTo(p.x, p.y);
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (st.rays) {
      const r = st.robots[0];
      st.rays.forEach((d, i) => {
        const a = r.th + RL_RAYS[i];
        ctx.strokeStyle = d < 0.3 ? T.bad : d < 0.7 ? T.warn : T.good; ctx.lineWidth = 2 * px;
        ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.x + Math.cos(a) * Math.min(d, 2), r.y + Math.sin(a) * Math.min(d, 2)); ctx.stroke();
      });
    }
    if (st.cat) drawCat(ctx, st.cat, px);
    st.robots.forEach((r, i) => drawRobot(ctx, r, px, T, i === 0 && st.showFov));
  }
  // טקסט בפיקסלים של המסך, לא ביחידות עולם
  label(text, wx, wy, size, color) {
    const ctx = this.ctx, w = this.world, dpr = this.cv.width / (this.cv.clientWidth || this.cv.width);
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.font = `${size * dpr}px Heebo, sans-serif`; ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, (wx - w.x0) * this.s, (wy - w.y0) * this.s);
    ctx.restore();
  }
  drawGrid(g, T) {
    const c = this.grid;
    if (c.width !== g.nx) { c.width = g.nx; c.height = g.ny; this.gv = -1; }
    if (this.gv !== g.version) {
      const cx = c.getContext('2d'); const img = cx.createImageData(g.nx, g.ny);
      const un = T.dark ? [40, 44, 52] : [150, 156, 166];
      for (let k = 0; k < g.lo.length; k++) {
        const v = g.lo[k], o = k * 4;
        if (v > 0.3) { img.data[o] = T.dark ? 230 : 30; img.data[o + 1] = T.dark ? 230 : 34; img.data[o + 2] = T.dark ? 235 : 40; img.data[o + 3] = 255; }
        else if (v < -0.3) { img.data[o + 3] = 0; }
        else { img.data[o] = un[0]; img.data[o + 1] = un[1]; img.data[o + 2] = un[2]; img.data[o + 3] = 235; }
      }
      cx.putImageData(img, 0, 0); this.gv = g.version;
    }
    const ctx = this.ctx; ctx.imageSmoothingEnabled = false;
    ctx.drawImage(c, g.x0, g.y0, g.nx * g.res, g.ny * g.res);
  }
}

function drawCat(ctx, c, px) {
  ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.th);
  ctx.fillStyle = '#e8913a';
  ctx.beginPath(); ctx.ellipse(0, 0, 0.17, 0.08, 0, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(0.17, 0, 0.065, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#e8913a'; ctx.lineWidth = 0.03; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(-0.16, 0); ctx.quadraticCurveTo(-0.3, 0.1, -0.33, -0.05); ctx.stroke();
  ctx.restore();
}

function drawRobot(ctx, r, px, T, fov) {
  ctx.save(); ctx.translate(r.x, r.y);
  if (fov) {
    ctx.fillStyle = T.accent; ctx.globalAlpha = 0.12;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, 2.2, r.th - FOV / 2, r.th + FOV / 2); ctx.closePath(); ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.rotate(r.th);
  // זחלים / גלגלים
  ctx.fillStyle = T.dark ? '#0b0d10' : '#22262c';
  ctx.fillRect(-0.09, -0.105, 0.18, 0.04); ctx.fillRect(-0.09, 0.065, 0.18, 0.04);
  ctx.fillStyle = r.color;
  ctx.beginPath(); ctx.roundRect(-0.085, -0.07, 0.17, 0.14, 0.03); ctx.fill();
  // הטלפון על התושבת (מבט מלמעלה: קו דק בחזית)
  if (r.phoneOn) {
    ctx.fillStyle = '#111'; ctx.fillRect(0.04, -0.08, 0.012, 0.16);
    ctx.fillStyle = '#5fd0ff'; ctx.fillRect(0.052, -0.07, 0.004, 0.14);
  }
  ctx.restore();
  if (!r.phoneOn) {
    // הטלפון שוכב על הרצפה מאחור
    ctx.save(); ctx.translate(r.x - Math.cos(r.th) * 0.2, r.y - Math.sin(r.th) * 0.2); ctx.rotate(r.th + 0.6);
    ctx.fillStyle = '#111'; ctx.fillRect(-0.08, -0.04, 0.16, 0.08);
    ctx.strokeStyle = T.bad; ctx.lineWidth = 2 * px; ctx.strokeRect(-0.08, -0.04, 0.16, 0.08);
    ctx.restore();
  }
  if (r.cutoff > 0) {
    ctx.fillStyle = T.bad; ctx.font = `${18 * px}px sans-serif`; ctx.textAlign = 'center';
    ctx.fillText('⚡', r.x, r.y - 0.18);
  }
}

// ---------- מבט המצלמה ----------
const turbo = t => {
  t = Math.max(0, Math.min(1, t));
  const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
  const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
  const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
  return [r, g, b].map(x => Math.max(0, Math.min(255, x)));
};

export class CamView {
  constructor(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.W = canvas.width; this.H = canvas.height;
    this.cols = this.W / 2; this.z = new Float32Array(this.cols);
  }
  draw(world, pose, opts) {
    const { ctx, W, H } = this;
    const camH = opts.camH, depth = opts.depth;
    const hor = H / 2, f = (W / 2) / Math.tan(FOV / 2);
    if (opts.fallen) {
      ctx.fillStyle = '#0a0a0a'; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#ff6b6b'; ctx.font = 'bold 20px Heebo'; ctx.textAlign = 'center'; ctx.direction = 'rtl';
      ctx.fillText('הטלפון על הרצפה, המצלמה רואה תקרה', W / 2, H / 2);
      return;
    }
    // שמיים/תקרה ורצפה
    let gr = ctx.createLinearGradient(0, 0, 0, hor);
    gr.addColorStop(0, depth ? '#30123b' : '#b9b4ab'); gr.addColorStop(1, depth ? '#30123b' : '#ddd8cf');
    ctx.fillStyle = gr; ctx.fillRect(0, 0, W, hor);
    gr = ctx.createLinearGradient(0, hor, 0, H);
    gr.addColorStop(0, depth ? '#4a0c6b' : '#8e7357'); gr.addColorStop(1, depth ? '#f9a03f' : '#c9a57f');
    ctx.fillStyle = gr; ctx.fillRect(0, hor, W, H - hor);

    const isArena = world.kind === 'arena';
    for (let c = 0; c < this.cols; c++) {
      const a = pose.th - FOV / 2 + FOV * (c + 0.5) / this.cols;
      const dx = Math.cos(a), dy = Math.sin(a), cosr = Math.cos(a - pose.th);
      this.z[c] = 1e9;
      if (isArena) {
        const pd = pose.x * dx + pose.y * dy, pp = pose.x * pose.x + pose.y * pose.y;
        const disc = pd * pd - (pp - world.R * world.R);
        if (disc > 0) {
          const t = -pd + Math.sqrt(disc);
          const yEdge = hor + camH * f / (t * cosr);
          ctx.fillStyle = depth ? '#30123b' : '#3a3f47';
          ctx.fillRect(c * 2, hor, 2, Math.max(0, yEdge - hor));
          if (!depth) { ctx.fillStyle = '#d8c3a5'; ctx.fillRect(c * 2, yEdge - 1, 2, 2); }
        }
        continue;
      }
      const hits = rayAll(world.segs, pose.x, pose.y, dx, dy, 14);
      for (const h of hits) {
        const p = h.d * cosr;
        const yT = hor - (h.seg.h - camH) * f / p, yB = hor + camH * f / p;
        let col;
        if (depth) col = turbo(1 - Math.min(1, p / 7));
        else {
          const sx = h.seg.x2 - h.seg.x1, sy = h.seg.y2 - h.seg.y1, L = Math.hypot(sx, sy) || 1;
          const face = 0.72 + 0.28 * Math.abs(sy / L);
          const k = face / (1 + 0.12 * p);
          col = h.seg.color.map(v => v * k);
        }
        ctx.fillStyle = `rgb(${col[0] | 0},${col[1] | 0},${col[2] | 0})`;
        ctx.fillRect(c * 2, yT, 2, yB - yT + 1);
        this.z[c] = Math.min(this.z[c], p);
      }
    }
    // ספרייטים: חתול או רובוט יריב
    for (const sp of opts.sprites || []) this.sprite(sp, pose, camH, f, hor, depth);
    if (opts.detect) this.box(opts.detect);
    if (opts.delay > 0.05) {
      ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(0, 0, W, 22);
      ctx.fillStyle = '#ffd36b'; ctx.font = '13px Heebo'; ctx.textAlign = 'right'; ctx.direction = 'rtl';
      ctx.fillText(`🔴 שידור חי, השהיה ${Math.round(opts.delay * 1000)}ms`, W - 8, 15);
    }
  }
  sprite(sp, pose, camH, f, hor, depth) {
    const dx = sp.x - pose.x, dy = sp.y - pose.y;
    const d = Math.hypot(dx, dy);
    const b = Math.atan2(dy, dx) - pose.th;
    const bb = Math.atan2(Math.sin(b), Math.cos(b));
    if (Math.abs(bb) > FOV / 2 + 0.3 || d < 0.08) return;
    const p = d * Math.cos(bb);
    if (p < 0.05) return;
    const sx = this.W / 2 + Math.tan(bb) * f;
    const col = Math.max(0, Math.min(this.cols - 1, (sx / 2) | 0));
    if (this.z[col] < p) return;
    const yB = hor + camH * f / p, yT = hor - (sp.h - camH) * f / p, w = sp.w * f / p;
    const ctx = this.ctx;
    sp.screen = { x: sx - w / 2, y: yT, w, h: yB - yT };
    if (depth) { const c = turbo(1 - Math.min(1, p / 7)); ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; ctx.fillRect(sx - w / 2, yT, w, yB - yT); return; }
    if (sp.kind === 'cat') {
      const hh = yB - yT;
      ctx.fillStyle = '#e8913a';
      ctx.beginPath(); ctx.ellipse(sx, yB - hh * 0.35, w * 0.45, hh * 0.32, 0, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(sx + w * 0.18, yB - hh * 0.72, hh * 0.22, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath();
      ctx.moveTo(sx + w * 0.05, yB - hh * 0.85); ctx.lineTo(sx + w * 0.1, yT - hh * 0.05); ctx.lineTo(sx + w * 0.18, yB - hh * 0.9);
      ctx.moveTo(sx + w * 0.2, yB - hh * 0.9); ctx.lineTo(sx + w * 0.3, yT - hh * 0.05); ctx.lineTo(sx + w * 0.33, yB - hh * 0.8);
      ctx.fill();
      ctx.fillStyle = '#222'; ctx.beginPath(); ctx.arc(sx + w * 0.12, yB - hh * 0.74, hh * 0.035, 0, 7); ctx.arc(sx + w * 0.25, yB - hh * 0.74, hh * 0.035, 0, 7); ctx.fill();
    } else {
      const hh = yB - yT;
      ctx.fillStyle = '#16181c'; ctx.fillRect(sx - w / 2, yB - hh * 0.25, w, hh * 0.25);
      ctx.fillStyle = sp.color; ctx.fillRect(sx - w * 0.42, yB - hh * 0.45, w * 0.84, hh * 0.25);
      if (sp.phoneOn) {
        ctx.fillStyle = '#111'; ctx.fillRect(sx - w * 0.38, yT, w * 0.76, hh * 0.55);
        ctx.fillStyle = '#5fd0ff'; ctx.fillRect(sx - w * 0.33, yT + hh * 0.05, w * 0.66, hh * 0.45);
        ctx.fillStyle = '#111'; const e = hh * 0.07;
        ctx.fillRect(sx - w * 0.18, yT + hh * 0.18, e, e * 1.6); ctx.fillRect(sx + w * 0.12, yT + hh * 0.18, e, e * 1.6);
      }
    }
  }
  box(det) {
    const s = det.sprite.screen; if (!s) return;
    const ctx = this.ctx;
    ctx.strokeStyle = '#34ff9a'; ctx.lineWidth = 2; ctx.strokeRect(s.x - 3, s.y - 3, s.w + 6, s.h + 6);
    ctx.fillStyle = '#34ff9a'; ctx.font = '12px Heebo'; ctx.textAlign = 'left'; ctx.direction = 'ltr';
    ctx.fillText(`${det.label} ${det.conf.toFixed(2)} · ${det.d.toFixed(1)}m`, s.x - 3, s.y - 7);
  }
}

// ---------- פרצוף ----------
export class Face {
  constructor(canvas) { this.cv = canvas; this.ctx = canvas.getContext('2d'); this.blink = 0; this.next = 2; this.look = 0; }
  draw(dt, mood, lookTo) {
    const { ctx } = this, W = this.cv.width, H = this.cv.height;
    this.next -= dt; if (this.next < 0) { this.blink = 0.15; this.next = 2 + Math.random() * 4; }
    this.blink = Math.max(0, this.blink - dt);
    this.look += ((lookTo || 0) - this.look) * Math.min(1, dt * 6);
    ctx.fillStyle = '#05070a'; ctx.fillRect(0, 0, W, H);
    const col = mood === 'angry' ? '#ff6b6b' : mood === 'sleepy' ? '#7a8496' : '#34e3c9';
    ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 5; ctx.lineCap = 'round';
    const ex = [W * 0.32, W * 0.68], ey = H * 0.45, ew = W * 0.16, eh = H * 0.42;
    for (const [i, x] of ex.entries()) {
      const lx = x + this.look * W * 0.08;
      if (mood === 'fallen') {
        ctx.beginPath(); ctx.moveTo(lx - 14, ey - 14); ctx.lineTo(lx + 14, ey + 14); ctx.moveTo(lx + 14, ey - 14); ctx.lineTo(lx - 14, ey + 14); ctx.stroke();
        continue;
      }
      let h = eh * (this.blink > 0 ? 0.1 : mood === 'sleepy' ? 0.25 : mood === 'scared' ? 1.15 : 1);
      ctx.beginPath(); ctx.roundRect(lx - ew / 2, ey - h / 2, ew, h, 10); ctx.fill();
      if (mood === 'angry') {
        ctx.fillStyle = '#05070a';
        ctx.beginPath();
        if (i === 0) { ctx.moveTo(lx - ew, ey - h / 2 - 4); ctx.lineTo(lx + ew, ey - h / 2 - 4); ctx.lineTo(lx + ew, ey - h * 0.05); }
        else { ctx.moveTo(lx - ew, ey - h / 2 - 4); ctx.lineTo(lx + ew, ey - h / 2 - 4); ctx.lineTo(lx - ew, ey - h * 0.05); }
        ctx.fill(); ctx.fillStyle = col;
      }
    }
    ctx.beginPath();
    if (mood === 'happy') ctx.arc(W / 2 + this.look * W * 0.05, H * 0.78, 14, 0.15 * Math.PI, 0.85 * Math.PI);
    else if (mood === 'scared' || mood === 'fallen') ctx.arc(W / 2, H * 0.86, 9, 0, Math.PI * 2);
    else { ctx.moveTo(W / 2 - 12, H * 0.84); ctx.lineTo(W / 2 + 12, H * 0.84); }
    ctx.stroke();
  }
}
