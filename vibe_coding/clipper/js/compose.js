/* compose.js: איך נראה פריים אחד של התוצאה.
 * אותה פונקציה מציירת גם את התצוגה המקדימה וגם את הייצוא, כך שמה שרואים
 * במסך הוא בדיוק מה שיוצא בקובץ.
 */
window.C = window.C || {};

C.compose = (() => {
  const RATIOS = { '9:16': [9, 16], '1:1': [1, 1], '4:5': [4, 5], '16:9': [16, 9] };

  /** גודל הפלט בפיקסלים. short = הצלע הקצרה (1080 או 720) */
  function outputSize(format, srcW, srcH, short) {
    let w, h;
    if (format === 'source' || !RATIOS[format]) {
      w = srcW || 1920; h = srcH || 1080;
    } else {
      [w, h] = RATIOS[format];
    }
    const k = short / Math.min(w, h);
    // קודקי H.264 דורשים מידות זוגיות
    const even = (x) => Math.max(2, Math.round(x / 2) * 2);
    return { W: even(w * k), H: even(h * k) };
  }

  // קנבס קטנטן לטשטוש זול: מקטינים לרוחב 24 פיקסלים ומגדילים בחזרה עם החלקה.
  // עובד בכל דפדפן (ctx.filter לא נתמך בספארי) ומהיר מספיק לכל פריים.
  const tiny = document.createElement('canvas');
  const tctx = tiny.getContext('2d');

  function srcDims(src) {
    return src.videoWidth ? [src.videoWidth, src.videoHeight] : [src.naturalWidth || src.width, src.naturalHeight || src.height];
  }

  function drawCover(ctx, src, W, H) {
    const [sw, sh] = srcDims(src);
    const k = Math.max(W / sw, H / sh);
    ctx.drawImage(src, (W - sw * k) / 2, (H - sh * k) / 2, sw * k, sh * k);
  }

  /**
   * איפה התמונה יושבת בתוך המסגרת.
   * zoom 1 = כל התמונה נראית (הצלע הארוכה נוגעת בקצוות); מעל 1 = זום פנימה וחיתוך.
   * px/py 0..1 = איפה התמונה בתוך המרווח: 0 = צמודה לשמאל/למעלה, 1 = לימין/למטה.
   * כשהתמונה גדולה מהמסגרת המרווח שלילי, ואז אותו מספר קובע איזה חלק נחתך.
   */
  function place(src, W, H, o) {
    const [sw, sh] = srcDims(src);
    const k = Math.min(W / sw, H / sh) * (o.zoom ?? 1);
    const w = sw * k, h = sh * k;
    return { x: (W - w) * (o.px ?? 0.5), y: (H - h) * (o.py ?? 0.5), w, h };
  }

  /** זום שבו התמונה ממלאת את כל המסגרת בלי שוליים */
  function coverZoom(src, W, H) {
    const [sw, sh] = srcDims(src);
    if (!sw) return 1;
    return Math.max(W / sw, H / sh) / Math.min(W / sw, H / sh);
  }

  function drawBlurBg(ctx, src, W, H) {
    const tw = 24, th = Math.max(2, Math.round((24 * H) / W));
    if (tiny.width !== tw || tiny.height !== th) { tiny.width = tw; tiny.height = th; }
    tctx.imageSmoothingEnabled = true;
    drawCover(tctx, src, tw, th);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tiny, 0, 0, W, H);
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }

  /**
   * מצייר פריים אחד.
   * @param src   וידאו (או תמונה) לצייר
   * @param o     { zoom, px, py, fill ('blur' | 'color'), fillColor, text: { top, bottom, size, color } או null,
   *                fade (0..1, 1 = שקוף לגמרי לשחור), logo, logoOn, corner, logoSize, logoOpacity, safe }
   */
  function draw(ctx, src, W, H, o) {
    ctx.fillStyle = o.fill === 'color' ? (o.fillColor || '#000') : '#000';
    ctx.fillRect(0, 0, W, H);
    let r = null;
    if (src && srcDims(src)[0]) {
      r = place(src, W, H, o);
      const gap = r.x > 0.5 || r.y > 0.5 || r.x + r.w < W - 0.5 || r.y + r.h < H - 0.5;
      if (gap && o.fill !== 'color') drawBlurBg(ctx, src, W, H);
      ctx.drawImage(src, r.x, r.y, r.w, r.h);
    }
    if (o.text) drawText(ctx, W, H, r, o);
    if (o.logoOn && o.logo) drawLogo(ctx, o.logo, W, H, o);
    if (o.fade > 0) {
      ctx.fillStyle = `rgba(0,0,0,${C.clamp(o.fade, 0, 1)})`;
      ctx.fillRect(0, 0, W, H);
    }
  }

  /** שובר טקסט לשורות שנכנסות ברוחב maxW. שורות חדשות של המשתמש נשמרות. */
  function wrap(ctx, text, maxW) {
    const out = [];
    for (const para of text.split('\n')) {
      let line = '';
      for (const word of para.split(/\s+/).filter(Boolean)) {
        const tryLine = line ? line + ' ' + word : word;
        if (line && ctx.measureText(tryLine).width > maxW) { out.push(line); line = word; }
        else line = tryLine;
      }
      out.push(line);
    }
    while (out.length && !out[out.length - 1]) out.pop();
    return out;
  }

  /**
   * טקסט ברצועה העליונה ובתחתונה. הבלוק מתמרכז בשטח הריק שבין קצה האזור הבטוח
   * לתמונה. אם אין שם מספיק מקום (זום גדול), הוא יושב על התמונה בקצה האזור הבטוח.
   */
  function drawText(ctx, W, H, r, o) {
    const tx = o.text;
    const z = o.safe || { top: 0, bottom: 0, left: 0, right: 0 };
    const m = Math.min(W, H) * 0.04;
    const left = Math.max(m, z.left * W), right = W - Math.max(m, z.right * W);
    const fs = Math.round((Math.min(W, H) * (tx.size ?? 7)) / 100);
    const lh = fs * 1.2;
    ctx.save();
    ctx.font = `700 ${fs}px "Segoe UI", Arial, "Noto Sans Hebrew", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    const safeTop = Math.max(m, z.top * H), safeBot = H - Math.max(m, z.bottom * H);
    for (const which of ['top', 'bottom']) {
      const raw = (tx[which] || '').trim();
      if (!raw) continue;
      ctx.direction = /[\u0590-\u05FF\u0600-\u06FF]/.test(raw) ? 'rtl' : 'ltr';
      const lines = wrap(ctx, raw, right - left);
      const bh = lines.length * lh;
      let y0;
      if (which === 'top') {
        const bandEnd = r ? Math.min(r.y, safeBot) : safeBot;
        y0 = bandEnd - safeTop >= bh + fs * 0.4 ? (safeTop + bandEnd - bh) / 2 : safeTop;
      } else {
        const bandStart = r ? Math.max(r.y + r.h, safeTop) : safeTop;
        y0 = safeBot - bandStart >= bh + fs * 0.4 ? (bandStart + safeBot - bh) / 2 : safeBot - bh;
      }
      lines.forEach((ln, i) => {
        const y = y0 + lh * (i + 0.5);
        ctx.lineWidth = fs * 0.14;
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.strokeText(ln, (left + right) / 2, y);
        ctx.fillStyle = tx.color || '#fff';
        ctx.fillText(ln, (left + right) / 2, y);
      });
    }
    ctx.restore();
  }

  function drawLogo(ctx, img, W, H, o) {
    const [iw, ih] = srcDims(img);
    if (!iw) return;
    // הגודל נמדד ביחס לצלע הקצרה, כך שהלוגו נראה אותו דבר ב-9:16 וב-16:9
    const short = Math.min(W, H);
    const lw = (short * (o.logoSize ?? 18)) / 100;
    const lh = (lw * ih) / iw;
    // הלוגו יושב בתוך האזור הבטוח של הפלטפורמה, כדי שהכפתורים והכיתוב של האפליקציה לא יכסו אותו
    const m = short * 0.04;
    const z = o.safe || { top: 0, bottom: 0, left: 0, right: 0 };
    const c = o.corner || 'tr';
    const x = c[1] === 'l' ? Math.max(m, z.left * W) : W - Math.max(m, z.right * W) - lw;
    const y = c[0] === 't' ? Math.max(m, z.top * H) : H - Math.max(m, z.bottom * H) - lh;
    ctx.save();
    ctx.globalAlpha = o.logoOpacity ?? 0.9;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, x, y, lw, lh);
    ctx.restore();
  }

  return { outputSize, draw, place, coverZoom };
})();
