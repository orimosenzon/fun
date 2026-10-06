/* sequencer.js: מנגן רצף של חלקים (פתיחה → קטע → סיום) לתוך קנבס, ואופציונלית מקליט.
 *
 * השיטה כמו ב-vedit: מנגנים בזמן אמת, מציירים כל פריים לקנבס, ומקליטים את הקנבס
 * (captureStream) יחד עם הקול ב-MediaRecorder. בלי ספריות ובלי שרת; המחיר הוא
 * שהייצוא אורך בערך כאורך התוצאה.
 *
 * הקול עובר דרך WebAudio: כל <video> מחובר ל-GainNode משלו (בשביל דהייה), ומשם
 * גם לרמקולים וגם ליעד ההקלטה. ככה הקול והתמונה של אותו וידאו נשארים מסונכרנים.
 */
window.C = window.C || {};

C.audio = (() => {
  let ac = null, master, monitor, recDest;
  let volume = 1;
  const nodes = new Map();

  function ensure() {
    if (!ac) {
      ac = new (window.AudioContext || window.webkitAudioContext)();
      master = ac.createGain();
      // העוצמה וההשתקה משפיעות רק על הרמקולים, לא על הקובץ המיוצא
      monitor = ac.createGain();
      monitor.gain.value = volume;
      master.connect(monitor);
      monitor.connect(ac.destination);
      recDest = ac.createMediaStreamDestination();
      master.connect(recDest);
    }
    if (ac.state === 'suspended') ac.resume();
    return ac;
  }

  /** מחבר וידאו לגרף הקול. אפשר לקרוא רק פעם אחת לכל אלמנט, ולכן שומרים במפה. */
  function gainFor(video) {
    ensure();
    if (!nodes.has(video)) {
      const src = ac.createMediaElementSource(video);
      const g = ac.createGain();
      src.connect(g);
      g.connect(master);
      nodes.set(video, g);
    }
    return nodes.get(video);
  }

  // דפדפנים משעים AudioContext שנוצר בלי לחיצה של המשתמש. מחדשים בכל לחיצה,
  // אחרת הווידאו מתנגן בלי קול (הקול שלו כבר מנותב דרך ההקשר המושעה).
  ['pointerdown', 'keydown'].forEach((ev) => document.addEventListener(ev, () => {
    if (ac && ac.state !== 'running') ac.resume();
  }, true));

  return {
    ensure,
    gainFor,
    setVolume(v) { volume = v; if (monitor) monitor.gain.value = v; },
    get ctx() { return ac; },
    recordTrack() { ensure(); return recDest.stream.getAudioTracks()[0]; },
  };
})();

C.sequencer = (() => {
  // לפי סדר תאימות. H.264+AAC בתוך MP4 נפתח בכל מקום ומתקבל בכל הרשתות.
  // בכרום על לינוקס אין AAC, ואז נופלים ל-Opus בתוך MP4 (ראו vedit/memory).
  const CANDIDATES = [
    { mime: 'video/mp4;codecs=avc1.640028,mp4a.40.2', ext: 'mp4', label: 'MP4 · H.264 + AAC' },
    { mime: 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', ext: 'mp4', label: 'MP4 · H.264 + AAC' },
    { mime: 'video/mp4;codecs=avc1.640028,opus', ext: 'mp4', label: 'MP4 · H.264 + Opus', warn: 'opusWarn' },
    { mime: 'video/mp4;codecs=avc1.42E01E,opus', ext: 'mp4', label: 'MP4 · H.264 + Opus', warn: 'opusWarn' },
    { mime: 'video/webm;codecs=vp9,opus', ext: 'webm', label: 'WebM · VP9', warn: 'webmWarn' },
    { mime: 'video/webm;codecs=vp8,opus', ext: 'webm', label: 'WebM · VP8', warn: 'webmWarn' },
    { mime: 'video/webm', ext: 'webm', label: 'WebM', warn: 'webmWarn' },
  ];

  function pickFormat() {
    if (typeof MediaRecorder === 'undefined') return null;
    return CANDIDATES.find((c) => { try { return MediaRecorder.isTypeSupported(c.mime); } catch { return false; } }) || null;
  }

  const FADE = 0.4;

  // השעון של הלולאה מגיע מ-Worker. בלשונית ברקע כרום מאט את requestAnimationFrame ואת
  // setTimeout לפעם בשנייה, ואחרי חמש דקות לפעם בדקה. אז התמונה קופאת, והעצירה בסוף
  // הקטע מאחרת: הקובץ יוצא ארוך עם שקט בסוף (ככה זה קרה ליורם). טיימר בתוך Worker לא מואט.
  // כל המתנה בזמן הקלטה עוברת דרכו, גם ההמתנה הקצרה לפני סגירת הקובץ: setTimeout של 60ms
  // הפך שם לשנייה, ואחרי חמש דקות ברקע לדקה שלמה של שקט בסוף הסרטון.
  const worker = (() => {
    try {
      const code = 'onmessage=(e)=>setTimeout(()=>postMessage(e.data.id),e.data.ms);';
      return new Worker(URL.createObjectURL(new Blob([code], { type: 'text/javascript' })));
    } catch { return null; }
  })();
  const timers = new Map();
  let timerId = 0;
  if (worker) worker.onmessage = (e) => { const fn = timers.get(e.data); timers.delete(e.data); fn?.(); };
  const after = (ms, fn) => {
    if (!worker) { setTimeout(fn, ms); return; }
    timers.set(++timerId, fn);
    worker.postMessage({ id: timerId, ms });
  };
  const sleep = (ms) => new Promise((r) => after(ms, r));
  const nextTick = (fn) => after(1000 / 60, fn);

  /**
   * @param o {
   *   parts:   [{ video, from, to, logo, label }]
   *   canvas:  קנבס היעד (בגודל הפלט)
   *   look:    אפשרויות ל-C.compose.draw (fit, focus, logo, corner, ...)
   *   fades:   דהייה בין חלקים
   *   record:  { mime, bitrate } או null לניגון בלבד
   *   onFrame(canvas), onProgress(frac, label)
   * }
   * @returns { done: Promise<Blob|null>, abort() }
   */
  function run(o) {
    let aborted = false;
    let rec = null;
    const ctx = o.canvas.getContext('2d', { alpha: false });
    const W = o.canvas.width, H = o.canvas.height;
    const total = o.parts.reduce((s, p) => s + (p.to - p.from), 0);
    const last = o.parts.length - 1;
    let doneBefore = 0;
    let current = null;
    const stats = [];

    const drawPart = (p, i, t) => {
      let fade = 0;
      if (o.fades) {
        if (i > 0) fade = Math.max(fade, 1 - (t - p.from) / FADE);
        if (i < last) fade = Math.max(fade, 1 - (p.to - t) / FADE);
      }
      C.compose.draw(ctx, p.video, W, H, { ...o.look, ...p.look, logoOn: p.logo, fade: C.clamp(fade, 0, 1) });
      o.onFrame?.(o.canvas);
    };

    async function playPart(p, i) {
      const v = p.video;
      const g = C.audio.gainFor(v);
      await C.seekTo(v, p.from);
      if (aborted) return;
      drawPart(p, i, p.from);

      current = v;
      await v.play();
      // ההקלטה מושהית בין החלקים (seek והפעלה של וידאו חדש לוקחים עשיריות שנייה),
      // כך שהזמן הזה לא נכנס לקובץ בכלל, והמעבר בתוצאה חלק.
      if (rec && rec.state === 'paused') rec.resume();

      const ac = C.audio.ctx;
      const now = ac.currentTime;
      const len = p.to - p.from;
      g.gain.cancelScheduledValues(now);
      const fadeIn = o.fades && i > 0, fadeOut = o.fades && i < last;
      g.gain.setValueAtTime(fadeIn ? 0 : 1, now);
      if (fadeIn) g.gain.linearRampToValueAtTime(1, now + FADE);
      if (fadeOut && len > 2 * FADE) {
        g.gain.setValueAtTime(1, now + len - FADE);
        g.gain.linearRampToValueAtTime(0, now + len);
      }
      // רשת ביטחון: שעון האודיו מדויק גם כשה-JS מאחר במחשב עמוס, ולכן משתיקים
      // בדיוק בנקודת הסוף. בלי זה נכנס לקובץ קול מאחרי החיתוך.
      g.gain.setValueAtTime(0, now + len + 0.03);

      const perf = { maxGap: 0, ticks: 0 };
      await new Promise((res) => {
        // מדידה לאבחון: כמה פעמים בשנייה הלולאה רצה. מתחת ל-30 הסרטון יוצא קטוע.
        let lastMs = performance.now();
        const tick = () => {
          const ms = performance.now();
          perf.maxGap = Math.max(perf.maxGap, ms - lastMs); lastMs = ms; perf.ticks++;
          if (aborted) { res(); return; }
          const t = v.currentTime;
          // כשהמחשב עמוס הלולאה מאחרת, והווידאו כבר עבר את נקודת הסוף. לא מציירים
          // פריים כזה: עדיף פריים קפוא לרגע מאשר תוכן שהמשתמש חתך החוצה.
          if (t >= p.to - 0.01 || v.ended) { res(); return; }
          drawPart(p, i, t);
          o.onProgress?.(C.clamp((doneBefore + (t - p.from)) / total, 0, 1), p.label);
          nextTick(tick);
        };
        tick();
      });
      v.pause();
      if (rec && rec.state === 'recording') rec.pause();
      stats.push({ part: p.label, from: p.from, to: p.to, stoppedAt: v.currentTime, fps: Math.round(perf.ticks / len), maxGapMs: Math.round(perf.maxGap) });
      current = null;
      doneBefore += len;
    }

    const done = (async () => {
      C.audio.ensure();
      let chunks = [];
      if (o.record) {
        const stream = o.canvas.captureStream(30);
        const at = C.audio.recordTrack();
        if (at) stream.addTrack(at);
        // פריים ראשון לפני תחילת ההקלטה, כדי שהקובץ לא יתחיל בשחור
        const p0 = o.parts[0];
        await C.seekTo(p0.video, p0.from);
        C.compose.draw(ctx, p0.video, W, H, { ...o.look, ...p0.look, logoOn: p0.logo, fade: 0 });
        rec = new MediaRecorder(stream, {
          mimeType: o.record.mime,
          videoBitsPerSecond: o.record.bitrate,
          audioBitsPerSecond: 192000,
        });
        rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
        rec.start(1000);
        rec.pause();   // playPart מפעיל אותה ברגע שהווידאו באמת מתנגן
      }
      try {
        for (let i = 0; i <= last && !aborted; i++) {
          // מכינים את החלק הבא ברקע כדי שהמעבר יהיה מיידי
          const next = o.parts[i + 1];
          if (next && next.video !== o.parts[i].video) C.seekTo(next.video, next.from).catch(() => {});
          await playPart(o.parts[i], i);
        }
      } finally {
        if (current) current.pause();
        if (rec && rec.state !== 'inactive') {
          o.onProgress?.(1, null);
          if (rec.state === 'paused') rec.resume();
          await sleep(60);
          const stopped = new Promise((r) => { rec.onstop = r; });
          rec.stop();
          await stopped;
        }
      }
      if (aborted || !o.record) return null;
      return new Blob(chunks, { type: o.record.mime.split(';')[0] });
    })();

    return {
      done,
      stats,
      abort() { aborted = true; if (current) current.pause(); },
    };
  }

  return { run, pickFormat };
})();
