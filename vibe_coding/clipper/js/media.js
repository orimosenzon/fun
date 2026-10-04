/* media.js: מה שהטיימליין צריך לדעת על הסרטון: תמונות ממוזערות וצורת גל של הקול.
 * שניהם נבנים ברקע אחרי הטעינה ומדווחים כשיש עוד חלק, כדי שהטיימליין יתמלא בהדרגה.
 */
window.C = window.C || {};

C.media = (() => {
  let token = 0;   // כל טעינה חדשה מבטלת את הקודמת

  /** תמונות ממוזערות לאורך הסרטון. onThumb(index, time, bitmap) */
  async function buildThumbs(blob, duration, count, height, onThumb) {
    const my = ++token;
    const v = document.createElement('video');
    v.muted = true;
    v.preload = 'auto';
    v.playsInline = true;
    try { await C.loadVideo(v, blob); } catch (e) { console.warn('thumbs', e); return; }
    const w = Math.round((height * v.videoWidth) / v.videoHeight) || height;
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = height;
    const ctx = cv.getContext('2d');
    // סדר "חציה בינארית": קודם כמה פריימים פזורים על כל האורך, אחר כך ממלאים ביניהם.
    // ככה אחרי שנייה כבר רואים את כל הסרטון בקירוב.
    const order = [];
    const seen = new Set();
    for (let step = Math.pow(2, Math.ceil(Math.log2(count))); step >= 1; step /= 2) {
      for (let i = 0; i < count; i += step) if (!seen.has(i)) { seen.add(i); order.push(i); }
    }
    for (const i of order) {
      if (my !== token) break;
      const t = Math.min(duration - 0.05, ((i + 0.5) * duration) / count);
      try { await C.seekTo(v, t); } catch { continue; }
      ctx.drawImage(v, 0, 0, w, height);
      const bmp = await createImageBitmap(cv);
      if (my !== token) break;
      onThumb(i, t, bmp);
    }
    URL.revokeObjectURL(v._url);
    v.removeAttribute('src');
    v.load();
  }

  /** צורת גל: מקסימום עוצמה לכל 1/100 שנייה. קבצים ענקיים מדלגים כדי לא לחנוק את הזיכרון. */
  async function buildPeaks(blob) {
    if (blob.size > 600 * 1024 * 1024) return null;
    try {
      const buf = await blob.arrayBuffer();
      const ac = new OfflineAudioContext(1, 1, 8000);
      const audio = await ac.decodeAudioData(buf);
      const rate = 100;
      const n = Math.ceil(audio.duration * rate);
      const peaks = new Float32Array(n);
      const per = audio.sampleRate / rate;
      for (let ch = 0; ch < audio.numberOfChannels; ch++) {
        const d = audio.getChannelData(ch);
        for (let i = 0; i < n; i++) {
          let m = peaks[i];
          const a = Math.floor(i * per), b = Math.min(d.length, Math.floor((i + 1) * per));
          for (let j = a; j < b; j += 4) { const x = d[j] < 0 ? -d[j] : d[j]; if (x > m) m = x; }
          peaks[i] = m;
        }
      }
      return { rate, peaks };
    } catch (e) {
      // סרטון בלי פס קול, או קודק שהדפדפן לא מפענח כאודיו בלבד
      console.warn('peaks', e);
      return null;
    }
  }

  return { buildThumbs, buildPeaks };
})();
