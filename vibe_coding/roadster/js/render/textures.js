// מרקמים פרוצדורליים על קנבס: בטון, אספלט, קרקע יבשה של טקסס, סימונים ושלטים.
import * as THREE from 'three';

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

function noiseFill(ctx, w, h, base, amp, seed, blobs = 0, blobAmp = 0) {
  const r = rng(seed);
  ctx.fillStyle = base; ctx.fillRect(0, 0, w, h);
  const img = ctx.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (r() - 0.5) * amp;
    d[i] += n; d[i + 1] += n; d[i + 2] += n;
  }
  ctx.putImageData(img, 0, 0);
  for (let i = 0; i < blobs; i++) {
    const x = r() * w, y = r() * h, rad = 8 + r() * 60;
    const g = ctx.createRadialGradient(x, y, 0, x, y, rad);
    const a = (r() - 0.5) * blobAmp;
    const c = a > 0 ? `rgba(255,255,255,${a})` : `rgba(0,0,0,${-a})`;
    g.addColorStop(0, c); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g; ctx.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
}

const cache = {};
function tex(c, repeat = 1) {
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function concreteTexture() {
  if (cache.concrete) return cache.concrete;
  const c = canvas(512, 512), x = c.getContext('2d');
  noiseFill(x, 512, 512, '#9e9b93', 22, 7, 60, 0.07);
  // תפרי יציקה כל 6 מ' (המרקם מכסה 12 מ')
  x.strokeStyle = 'rgba(60,58,54,0.55)'; x.lineWidth = 2;
  for (let i = 0; i <= 512; i += 256) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i, 512); x.stroke(); x.beginPath(); x.moveTo(0, i); x.lineTo(512, i); x.stroke(); }
  // כתמי צמיגים
  const r = rng(3);
  for (let i = 0; i < 18; i++) {
    x.strokeStyle = `rgba(30,30,30,${0.05 + r() * 0.1})`; x.lineWidth = 6 + r() * 6;
    x.beginPath(); const sx = r() * 512, sy = r() * 512; x.moveTo(sx, sy);
    x.bezierCurveTo(sx + r() * 200 - 100, sy + r() * 200 - 100, sx + r() * 300 - 150, sy + r() * 300 - 150, sx + r() * 400 - 200, sy + r() * 400 - 200); x.stroke();
  }
  return (cache.concrete = tex(c));
}

export function asphaltTexture() {
  if (cache.asphalt) return cache.asphalt;
  const c = canvas(512, 512), x = c.getContext('2d');
  noiseFill(x, 512, 512, '#3a3a3c', 30, 11, 40, 0.06);
  const r = rng(5);
  for (let i = 0; i < 4000; i++) { x.fillStyle = `rgba(200,200,200,${r() * 0.12})`; x.fillRect(r() * 512, r() * 512, 1.5, 1.5); }
  return (cache.asphalt = tex(c));
}

export function dirtTexture() {
  if (cache.dirt) return cache.dirt;
  const c = canvas(1024, 1024), x = c.getContext('2d');
  noiseFill(x, 1024, 1024, '#9c8a64', 26, 21, 260, 0.16);
  const r = rng(9);
  // שיחים ועשב יבש
  for (let i = 0; i < 2600; i++) {
    const px = r() * 1024, py = r() * 1024, s = 2 + r() * 7;
    const g = 70 + r() * 50;
    x.fillStyle = `rgba(${g + 20},${g + 30},${g - 20},${0.25 + r() * 0.35})`;
    x.beginPath(); x.arc(px, py, s, 0, Math.PI * 2); x.fill();
  }
  return (cache.dirt = tex(c));
}

export function steelTexture() {
  if (cache.steel) return cache.steel;
  const c = canvas(256, 256), x = c.getContext('2d');
  noiseFill(x, 256, 256, '#8d9298', 14, 31, 20, 0.05);
  x.strokeStyle = 'rgba(40,40,45,0.35)'; x.lineWidth = 2;
  for (let i = 0; i <= 256; i += 64) { x.beginPath(); x.moveTo(0, i); x.lineTo(256, i); x.stroke(); }
  return (cache.steel = tex(c));
}

// משטח הרכיבה של הרמפות: פלדה צבועה עם פסי הדרכה
export function rampTexture(color = '#e9e6df', stripe = '#d32f2f') {
  const key = 'ramp' + color + stripe;
  if (cache[key]) return cache[key];
  const c = canvas(512, 512), x = c.getContext('2d');
  noiseFill(x, 512, 512, color, 12, 41, 30, 0.05);
  x.fillStyle = stripe;
  x.fillRect(0, 0, 18, 512); x.fillRect(494, 0, 18, 512);
  x.fillStyle = 'rgba(60,60,60,0.25)';
  for (let i = 0; i < 512; i += 64) x.fillRect(0, i, 512, 2);
  return (cache[key] = tex(c));
}

// שלט טקסט (תומך עברית)
export function labelTexture(text, { w = 1024, h = 256, bg = 'rgba(12,14,18,0.86)', fg = '#fff', font = 'bold 120px Rubik, Arial', sub = '', subFont = '60px Rubik, Arial' } = {}) {
  const c = canvas(w, h), x = c.getContext('2d');
  x.fillStyle = bg; x.fillRect(0, 0, w, h);
  x.fillStyle = fg; x.textAlign = 'center'; x.textBaseline = 'middle'; x.direction = 'rtl';
  x.font = font; x.fillText(text, w / 2, sub ? h * 0.38 : h / 2);
  if (sub) { x.font = subFont; x.globalAlpha = 0.8; x.fillText(sub, w / 2, h * 0.76); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

// כתם רך לחלקיקים
export function softDot() {
  if (cache.dot) return cache.dot;
  const c = canvas(64, 64), x = c.getContext('2d');
  const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.4, 'rgba(255,255,255,0.55)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 64, 64);
  return (cache.dot = new THREE.CanvasTexture(c));
}
