/* transitions.js: ציור המעברים בין שני קליפים
 *
 * כל מעבר מקבל שתי פונקציות ציור: drawA (הקליפ היוצא) ו-drawB (הנכנס),
 * ומתקדם לפי p בין 0 ל-1. הפונקציות מציירות ביחס למערכת הצירים הנוכחית,
 * כך שאפשר לעטוף אותן ב-translate/clip בלי לשבור אותן.
 */

const easeInOut = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);

export function drawTransition(ctx, type, p, drawA, drawB, w, h) {
  p = Math.max(0, Math.min(1, p));

  switch (type) {
    /* ── מעבר הדרגתי (cross dissolve) ── */
    case 'dissolve': {
      drawA(ctx);
      ctx.save(); ctx.globalAlpha = p; drawB(ctx); ctx.restore();
      break;
    }

    /* ── דרך שחור / לבן ── */
    case 'fadeblack':
    case 'fadewhite': {
      const color = type === 'fadewhite' ? '#fff' : '#000';
      if (p < 0.5) {
        drawA(ctx);
        ctx.save(); ctx.globalAlpha = p * 2; ctx.fillStyle = color; ctx.fillRect(0, 0, w, h); ctx.restore();
      } else {
        ctx.save(); ctx.fillStyle = color; ctx.fillRect(0, 0, w, h); ctx.restore();
        ctx.save(); ctx.globalAlpha = (p - 0.5) * 2; drawB(ctx); ctx.restore();
      }
      break;
    }

    /* ── מחיקות ── */
    case 'wipeleft':   wipe(ctx, p, drawA, drawB, w, h, 'l'); break;
    case 'wiperight':  wipe(ctx, p, drawA, drawB, w, h, 'r'); break;
    case 'wipeup':     wipe(ctx, p, drawA, drawB, w, h, 'u'); break;

    /* ── החלקה: הקליפ הנכנס נכנס מעל ── */
    case 'slideleft': {
      drawA(ctx);
      const e = easeInOut(p);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
      ctx.translate(w * (1 - e), 0);
      drawB(ctx);
      ctx.restore();
      break;
    }
    case 'slideup': {
      drawA(ctx);
      const e = easeInOut(p);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
      ctx.translate(0, h * (1 - e));
      drawB(ctx);
      ctx.restore();
      break;
    }

    /* ── דחיפה: שניהם זזים ── */
    case 'push': {
      const e = easeInOut(p);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
      ctx.save(); ctx.translate(-w * e, 0); drawA(ctx); ctx.restore();
      ctx.save(); ctx.translate(w * (1 - e), 0); drawB(ctx); ctx.restore();
      ctx.restore();
      break;
    }

    /* ── זום ── */
    case 'zoomin': {
      const e = easeInOut(p);
      ctx.save();
      ctx.translate(w / 2, h / 2); ctx.scale(1 + 0.25 * e, 1 + 0.25 * e); ctx.translate(-w / 2, -h / 2);
      ctx.globalAlpha = 1;
      drawA(ctx);
      ctx.restore();
      ctx.save();
      const s = 0.7 + 0.3 * e;
      ctx.globalAlpha = e;
      ctx.translate(w / 2, h / 2); ctx.scale(s, s); ctx.translate(-w / 2, -h / 2);
      drawB(ctx);
      ctx.restore();
      break;
    }

    /* ── עיגול מתרחב ── */
    case 'circle': {
      drawA(ctx);
      const r = easeInOut(p) * Math.hypot(w, h) / 2;
      ctx.save();
      ctx.beginPath(); ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2); ctx.clip();
      drawB(ctx);
      ctx.restore();
      break;
    }

    /* ── תריסים ── */
    case 'blinds': {
      drawA(ctx);
      const bands = 10, bh = h / bands;
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < bands; i++) ctx.rect(0, i * bh, w, bh * p);
      ctx.clip();
      drawB(ctx);
      ctx.restore();
      break;
    }

    default: {
      drawA(ctx);
      ctx.save(); ctx.globalAlpha = p; drawB(ctx); ctx.restore();
    }
  }
}

function wipe(ctx, p, drawA, drawB, w, h, dir) {
  drawA(ctx);
  const soft = Math.max(2, w * 0.012);
  ctx.save();
  ctx.beginPath();
  if (dir === 'l')      ctx.rect(0, 0, w * p, h);
  else if (dir === 'r') ctx.rect(w * (1 - p), 0, w * p, h);
  else                  ctx.rect(0, h * (1 - p), w, h * p);
  ctx.clip();
  drawB(ctx);
  ctx.restore();
  // קו רך על הגבול, כדי שהמעבר לא ייראה חד מדי
  if (p > 0.001 && p < 0.999) {
    ctx.save();
    let g;
    if (dir === 'l')      { g = ctx.createLinearGradient(w * p - soft, 0, w * p + soft, 0); }
    else if (dir === 'r') { g = ctx.createLinearGradient(w * (1 - p) + soft, 0, w * (1 - p) - soft, 0); }
    else                  { g = ctx.createLinearGradient(0, h * (1 - p) + soft, 0, h * (1 - p) - soft); }
    g.addColorStop(0, 'rgba(255,255,255,0)');
    g.addColorStop(0.5, 'rgba(255,255,255,.25)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }
}

/* ─────────────── תצוגה מקדימה בפאנל המעברים ─────────────── */

export function paintPreview(canvas, type, p = 0.55) {
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  const a = (c) => {
    const g = c.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, '#2e6fb0'); g.addColorStop(1, '#123a63');
    c.fillStyle = g; c.fillRect(0, 0, w, h);
    c.fillStyle = 'rgba(255,255,255,.75)'; c.font = `${Math.round(h * 0.4)}px sans-serif`;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText('A', w * 0.5, h * 0.5);
  };
  const b = (c) => {
    const g = c.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, '#c98b3a'); g.addColorStop(1, '#7d4b12');
    c.fillStyle = g; c.fillRect(0, 0, w, h);
    c.fillStyle = 'rgba(255,255,255,.8)'; c.font = `${Math.round(h * 0.4)}px sans-serif`;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText('B', w * 0.5, h * 0.5);
  };

  drawTransition(ctx, type, p, a, b, w, h);
}
