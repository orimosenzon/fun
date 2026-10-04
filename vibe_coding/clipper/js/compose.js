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

  function drawCover(ctx, src, W, H, focus) {
    const [sw, sh] = srcDims(src);
    const k = Math.max(W / sw, H / sh);
    const dw = sw * k, dh = sh * k;
    // focus 0..1 קובע איזה חלק נשאר כשחותכים: 0 = שמאל/למעלה, 1 = ימין/למטה
    const dx = (W - dw) * focus;
    const dy = (H - dh) * 0.5;
    ctx.drawImage(src, dx, dy, dw, dh);
  }

  function drawContain(ctx, src, W, H) {
    const [sw, sh] = srcDims(src);
    const k = Math.min(W / sw, H / sh);
    const dw = sw * k, dh = sh * k;
    ctx.drawImage(src, (W - dw) / 2, (H - dh) / 2, dw, dh);
  }

  function drawBlurBg(ctx, src, W, H) {
    const tw = 24, th = Math.max(2, Math.round((24 * H) / W));
    if (tiny.width !== tw || tiny.height !== th) { tiny.width = tw; tiny.height = th; }
    tctx.imageSmoothingEnabled = true;
    drawCover(tctx, src, tw, th, 0.5);
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
   * @param o     { fit, focus, fade (0..1, 1 = שקוף לגמרי לשחור), logo, logoOn, corner, logoSize, logoOpacity }
   */
  function draw(ctx, src, W, H, o) {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);
    if (src && srcDims(src)[0]) {
      if (o.fit === 'cover') drawCover(ctx, src, W, H, o.focus ?? 0.5);
      else {
        if (o.fit === 'blur') drawBlurBg(ctx, src, W, H);
        drawContain(ctx, src, W, H);
      }
    }
    if (o.logoOn && o.logo) drawLogo(ctx, o.logo, W, H, o);
    if (o.fade > 0) {
      ctx.fillStyle = `rgba(0,0,0,${C.clamp(o.fade, 0, 1)})`;
      ctx.fillRect(0, 0, W, H);
    }
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

  return { outputSize, draw };
})();
