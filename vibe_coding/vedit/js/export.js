/* export.js: ייצוא הסרט לקובץ
 *
 * השיטה: מריצים את הטיימליין בזמן אמת, מצלמים את הקנבס (captureStream) יחד עם
 * מיקס האודיו של המנוע, ומקליטים ב-MediaRecorder. זה עובד בכל דפדפן מודרני
 * בלי שום ספרייה חיצונית, והמחיר הוא שהייצוא אורך בערך כאורך הסרט.
 */

import { $, clamp, download, toast, shortDur, baseName } from './util.js';
import { state, proj, fps, duration } from './state.js';

/* לפי סדר תאימות: H.264+AAC נפתח בכל נגן ובכל תוכנת עריכה.
   מה שזמין בפועל תלוי בדפדפן ובמערכת ההפעלה. */
const CANDIDATES = [
  { mime: 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', label: 'MP4 · H.264 + AAC (הכי תואם)', ext: 'mp4' },
  { mime: 'video/mp4;codecs=avc1.42E01E,opus',      label: 'MP4 · H.264 + Opus', ext: 'mp4' },
  { mime: 'video/webm;codecs=vp9,opus',             label: 'WebM · VP9', ext: 'webm' },
  { mime: 'video/webm;codecs=vp8,opus',             label: 'WebM · VP8 (הכי בטוח)', ext: 'webm' },
  { mime: 'video/mp4',                              label: 'MP4 · ברירת המחדל של הדפדפן', ext: 'mp4' },
  { mime: 'video/webm',                             label: 'WebM · ברירת המחדל של הדפדפן', ext: 'webm' },
];

export function supportedFormats() {
  if (typeof MediaRecorder === 'undefined') return [];
  return CANDIDATES.filter((c) => {
    try { return MediaRecorder.isTypeSupported(c.mime); } catch { return false; }
  });
}

let running = null;

export function isExporting() { return !!running; }

/**
 * @param engine מנוע הניגון
 * @param opts { scale, bitrate, mime, ext, from, to, onProgress, onDone, onError }
 */
export async function exportMovie(engine, opts) {
  if (running) return;
  const total = duration();
  const from = clamp(opts.from ?? 0, 0, total);
  const to = clamp(opts.to ?? total, from + 0.05, total);
  if (to - from < 0.05) { opts.onError?.(new Error('אין מה לייצא: הטיימליין ריק')); return; }

  const W = Math.round((proj().width * opts.scale) / 2) * 2;
  const H = Math.round((proj().height * opts.scale) / 2) * 2;
  const rate = fps();

  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d', { alpha: false });
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);

  const stream = cv.captureStream(rate);
  let audioTracks = [];
  try {
    const a = engine.captureAudioStream();
    audioTracks = a.getAudioTracks();
    audioTracks.forEach((t) => stream.addTrack(t));
  } catch (e) { console.warn('no audio for export', e); }

  let rec;
  try {
    rec = new MediaRecorder(stream, {
      mimeType: opts.mime,
      videoBitsPerSecond: opts.bitrate,
      audioBitsPerSecond: 192000,
    });
  } catch (e) { opts.onError?.(e); return; }

  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };

  const finish = () => {
    const wasAborted = running?.aborted;
    running = null;
    engine.onTick = prevTick;
    engine.onEnd = prevEnd;
    engine.stopAt = null;
    if (wasAborted) { opts.onAbort?.(); return; }   // ביטול: לא מורידים קובץ חלקי
    const blob = new Blob(chunks, { type: opts.mime.split(';')[0] });
    opts.onDone?.(blob, opts.ext);
  };
  rec.onstop = finish;
  rec.onerror = (e) => { running = null; opts.onError?.(e.error || new Error('שגיאת הקלטה')); };

  const prevTick = engine.onTick;
  const prevEnd = engine.onEnd;

  const paint = (t) => {
    ctx.drawImage(engine.canvas, 0, 0, W, H);
    opts.onProgress?.(clamp((t - from) / (to - from), 0, 1), t);
  };

  engine.onTick = (t) => { paint(t); prevTick?.(t); };
  engine.onEnd = () => {
    // משאירים רגע קטן כדי שהפריים האחרון והאודיו ייכנסו
    setTimeout(() => { try { rec.stop(); } catch { finish(); } }, 350);
  };

  running = { rec, engine };

  // מתחילים מהנקודה המבוקשת
  engine.pause();
  state.playhead = from;
  engine.stopAt = to;
  await engine.waitReady(from);
  engine.seek(from, true);
  paint(from);

  rec.start(400);
  await engine.play();
}

export function abortExport() {
  if (!running) return;
  running.aborted = true;
  try { running.engine.pause(); } catch {}
  try { running.rec.stop(); } catch {}
}

/* ─────────── חיבור לחלון הייצוא ─────────── */

export function initExportUI(engine) {
  const modal = $('#modalExport');
  const sel = $('#expFormat');
  const note = $('#expNote');
  const formats = supportedFormats();

  if (!formats.length) {
    note.textContent = 'הדפדפן הזה לא תומך בהקלטת וידאו (MediaRecorder). נסו Chrome או Edge עדכניים.';
    $('#expStart').disabled = true;
  } else {
    formats.forEach((f, i) => {
      const o = document.createElement('option');
      o.value = String(i); o.textContent = f.label;
      sel.appendChild(o);
    });
  }

  const open = () => {
    if (duration() <= 0) { toast('הטיימליין ריק, אין מה לייצא', 'err'); return; }
    const hasRange = state.inPoint != null || state.outPoint != null;
    $('#expRange').disabled = !hasRange;
    $('#expRange').checked = hasRange;
    $('#expProgress').classList.add('hidden');
    $('#expStart').disabled = !formats.length;
    $('#expStart').textContent = 'התחל ייצוא';
    $('#expCancel').textContent = 'ביטול';
    updateNote();
    modal.classList.remove('hidden');
  };
  const close = () => modal.classList.add('hidden');

  function rangeNow() {
    const useRange = $('#expRange').checked && !$('#expRange').disabled;
    return {
      from: useRange ? (state.inPoint ?? 0) : 0,
      to: useRange ? (state.outPoint ?? duration()) : duration(),
    };
  }

  function updateNote() {
    const { from, to } = rangeNow();
    const scale = parseFloat($('#expRes').value);
    const W = Math.round(proj().width * scale), H = Math.round(proj().height * scale);
    note.textContent = `הפלט: ${W}×${H} ב-${fps()} fps, אורך ${shortDur(to - from)}. `
      + 'הרינדור מתבצע בזמן אמת, כלומר הוא יימשך בערך כאורך הסרט. '
      + 'השאירו את הלשונית פתוחה וגלויה עד שהוא נגמר.';
  }

  $('#expRes').addEventListener('change', updateNote);
  $('#expRange').addEventListener('change', updateNote);

  $('#btnExport').addEventListener('click', open);
  $('#expCancel').addEventListener('click', () => {
    if (isExporting()) { abortExport(); toast('הייצוא בוטל'); }
    close();
  });

  $('#expStart').addEventListener('click', async () => {
    if (isExporting()) return;
    const f = formats[parseInt(sel.value || '0', 10)];
    const { from, to } = rangeNow();
    $('#expProgress').classList.remove('hidden');
    $('#expStart').disabled = true;
    $('#expCancel').textContent = 'עצור';
    $('#expBar').style.width = '0%';
    $('#expStatus').textContent = 'מרנדר…';

    await exportMovie(engine, {
      scale: parseFloat($('#expRes').value),
      bitrate: parseInt($('#expQuality').value, 10),
      mime: f.mime, ext: f.ext, from, to,
      onProgress: (p, t) => {
        $('#expBar').style.width = `${(p * 100).toFixed(1)}%`;
        $('#expStatus').textContent = `מרנדר… ${Math.round(p * 100)}% (${shortDur(t - from)} מתוך ${shortDur(to - from)})`;
      },
      onDone: (blob, ext) => {
        $('#expBar').style.width = '100%';
        $('#expStatus').textContent = 'מוכן! הקובץ יורד…';
        const name = `${baseName(proj().name || 'vedit')}.${ext}`;
        download(blob, name);
        toast(`הייצוא הסתיים: ${name}`, 'ok', 5000);
        $('#expStart').disabled = false;
        $('#expCancel').textContent = 'סגור';
        setTimeout(close, 1200);
      },
      onAbort: () => {
        $('#expStatus').textContent = 'הייצוא בוטל.';
        $('#expStart').disabled = false;
        $('#expCancel').textContent = 'סגור';
      },
      onError: (err) => {
        console.error(err);
        $('#expStatus').textContent = `שגיאה בייצוא: ${err.message || err}`;
        $('#expStart').disabled = false;
        $('#expCancel').textContent = 'סגור';
      },
    });
  });
}
