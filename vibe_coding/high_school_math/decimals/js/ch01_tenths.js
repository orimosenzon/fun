'use strict';
/* פרק 1: עשיריות. טבלת שוקולד שמחולקת ל-10 חתיכות. */

// ציור של t עשיריות כטבלאות שוקולד בשורה
function barsSVG(t, opt = {}) {
  const n = Math.max(opt.min || 1, Math.ceil(t / 10)), s = opt.s || 120, gap = 22;
  const W = n * s + (n - 1) * gap + 12, H = s + 12;
  const svg = S.svg(W, H, 'board'); svg.style.maxWidth = Math.min(opt.maxW || W, W) + 'px';
  for (let b = 0; b < n; b++) drawBar(svg, 6 + b * (s + gap), 6, s, clamp(t - b * 10, 0, 10), opt.pick ? { onPick: i => opt.pick(b, i) } : {});
  return svg;
}
const tenthStr = t => N.str(N.of(t, 1), 1);

App.add({
  id: 'ch1', short: 'עשיריות', icon: '🍫', title: 'עשיריות: שוקולד לעשר',
  desc: 'חותכים שלם ל-10 חתיכות, ומגלים מה זה הנקודה',
  intro: `<p>הכל מתחיל מדבר אחד פשוט: לחתוך שלם ל-10 חלקים שווים. מזה בנוי כל העולם של השברים העשרוניים.</p>`,
  steps: [
    {
      t: 'טבלה אחת, עשר חתיכות',
      r(el, done) {
        el.innerHTML = `
          <p>הנה טבלת שוקולד. היא מחולקת ל-<b>10 חתיכות שוות</b>. כל חתיכה היא <b>עשירית</b> מהטבלה, כי צריך בדיוק 10 חתיכות כאלה כדי להרכיב טבלה שלמה.</p>
          <p class="do">לחץ על החתיכות כדי לקחת אותן. לחיצה על החתיכה האחרונה שלקחת מחזירה אותה.</p>
          <div class="play">
            <div><div class="bar-host"></div><div class="controls"><button class="btn sm clr">להחזיר הכל</button></div></div>
            <div class="panel readout"></div>
          </div>
          <div class="mhost"></div>
          <div class="after" hidden>
            <div class="aha"><b>מה ראית כאן?</b>
              <ul>
                <li>${M('0.3')} פירושו <b>3 עשיריות</b>: שלוש חתיכות מתוך עשר. זה בדיוק כמו השבר ${F(3, 10)}.</li>
                <li>ה-<b class="t-p0">0</b> לפני הנקודה אומר: <b>אין אף טבלה שלמה</b>. יש רק חתיכות.</li>
                <li>ה<b>נקודה</b> היא גבול: משמאל לה סופרים טבלאות שלמות, ומימין לה סופרים חתיכות.</li>
                <li>חצי טבלה זה 5 חתיכות מתוך 10, ולכן חצי = ${M('0.5')}.</li>
              </ul>
            </div>
          </div>`;
        let k = 0;
        const host = el.querySelector('.bar-host'), ro = el.querySelector('.readout');
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `קח ${M('0.4')} מהטבלה`, ok: s => s === 4 },
          { t: 'קח בדיוק חצי טבלה', ok: s => s === 5, after: () => note(`חצי טבלה = 5 חתיכות מתוך 10 = ${M('0.5')}`) },
          { t: 'קח שבע עשיריות', ok: s => s === 7 },
          { t: `קח ${M('0.1')}`, ok: s => s === 1 },
          { t: 'קח את כל הטבלה', ok: s => s === 10, after: () => note(`10 עשיריות הן טבלה שלמה! כותבים ${M('1')} (או ${M('1.0')}): טבלה שלמה אחת, ו-0 חתיכות נוספות.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function draw() {
          host.innerHTML = ''; const svg = S.svg(300, 300, 'board'); svg.style.maxWidth = '320px';
          drawBar(svg, 15, 15, 270, k, { onPick: i => { k = (k === i + 1) ? i : i + 1; draw(); mis.check(k); } });
          host.appendChild(svg);
          const dec = k === 10 ? '1.0' : '0.' + k;
          ro.innerHTML = `
            <div class="ro-row"><span class="ro-lbl">לקחת</span><b>${k} חתיכות מתוך 10</b></div>
            <div class="ro-row"><span class="ro-lbl">כשבר</span>${F(k, 10)}</div>
            <div class="ro-row"><span class="ro-lbl">במילים</span><span class="ro-words">${k === 0 ? 'אפס' : k === 10 ? 'עשר עשיריות = שלם אחד' : HEB.count(k, 'עשירית', 'עשיריות')}</span></div>
            <div class="ro-big"><div class="bignum">${digitsHTML(dec)}</div>
              <div class="place-tags"><span class="t-p0">שלמות</span><span>.</span><span class="t-pm1">חתיכות (עשיריות)</span></div></div>`;
        }
        el.querySelector('.clr').onclick = () => { k = 0; draw(); };
        draw();
      },
    },
    {
      t: 'יותר מטבלה אחת',
      r(el, done) {
        el.innerHTML = `
          <p>מה קורה כשיש לך יותר משוקולד אחד? כל פעם שמצטברות 10 חתיכות, יש לך עוד טבלה שלמה.</p>
          <p class="do">הוסף והורד עשיריות עם הכפתורים, ובצע את המשימות.</p>
          <div class="bars-host"></div>
          <div class="controls">
            <button class="btn round minus" title="הורד עשירית">−</button>
            <button class="btn round plus" title="הוסף עשירית">+</button>
            <span class="muted small">כל לחיצה = עשירית אחת (חתיכה אחת)</span>
            <button class="btn sm plus10">+ טבלה שלמה</button>
          </div>
          <div class="panel readout"></div>
          <div class="qhost"></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha"><b>הכלל החשוב של הפרק:</b> מימין לנקודה יש מקום לספרה אחת בלבד, מ-0 עד 9. כשמגיעים ל-10 עשיריות, הן הופכות לשלם אחד ועוברות לצד השני של הנקודה. בדיוק כמו במספרים רגילים: אחרי 9 בא 10, ולא "ספרה תשע-עשרה".</div></div>`;
        let t = 7, asked = false, blocked = false;
        const host = el.querySelector('.bars-host'), ro = el.querySelector('.readout');
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `הגע ל-${M('0.9')}`, ok: v => v === 9, after: askNext },
          { t: 'עכשיו הוסף עוד עשירית אחת, ותראה מה קורה', ok: v => v === 10, after: () => pop() },
          { t: `הגע ל-${M('1.6')}`, ok: v => v === 16, after: () => note(`שים לב: ב-${M('1.6')} יש בסך הכל 16 חתיכות. טבלה שלמה (10) ועוד 6.`) },
          { t: `הגע ל-${M('3')}`, ok: v => v === 30 },
          { t: `ועכשיו רד ל-${M('2.8')}`, ok: v => v === 28 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function pop() { const b = ro.querySelector('.bignum'); if (b) { b.classList.remove('pop'); void b.offsetWidth; b.classList.add('pop'); } }
        function askNext() {
          if (asked) return; asked = true; blocked = true;
          const q = el.querySelector('.qhost');
          q.innerHTML = `<div class="modal-q"><b>רגע, לפני שממשיכים!</b> יש לך עכשיו ${M('0.9')}. אם תוסיף עוד עשירית אחת, איזה מספר יהיה לך?
            <div class="opts" style="margin-top:8px">${['0.10', '1.0', '0.91'].map(o => `<button class="opt" data-v="${o}">${M(o)}</button>`).join('')}</div><div class="fbq"></div></div>`;
          q.querySelectorAll('.opt').forEach(b => b.onclick = () => {
            q.querySelectorAll('.opt').forEach(x => x.disabled = true);
            const ok = b.dataset.v === '1.0';
            b.classList.add(ok ? 'right' : 'wrong');
            if (!ok) q.querySelector('[data-v="1.0"]').classList.add('right');
            q.querySelector('.fbq').innerHTML = `<p>${ok ? '<b>נכון!</b> ' : `<b>הרבה חושבים ככה, אבל לא.</b> `}
              9 חתיכות ועוד חתיכה = 10 חתיכות. ו-10 חתיכות הן <b>טבלה שלמה</b>. אז יש לך טבלה אחת, ו-0 חתיכות נוספות: ${M('1.0')}, או פשוט ${M('1')}.
              ${!ok && b.dataset.v === '0.10' ? `(${M('0.10')} זה משהו אחר לגמרי. נגלה בפרק הבא שזה בכלל שווה ל-${M('0.1')}.)` : ''} עכשיו לחץ על + ותראה בעצמך.</p>`;
            blocked = false; update();
          });
          update();
        }
        function update() {
          host.innerHTML = ''; host.appendChild(barsSVG(t, { s: 130, min: 1, maxW: 700 }));
          const w = Math.floor(t / 10), p = t % 10;
          ro.innerHTML = `
            <div class="ro-row" style="justify-content:space-between">
              <div class="bignum">${digitsHTML(tenthStr(t))}</div>
              <div><div class="ro-words">${HEB.read(N.of(t, 1)) || 'אפס'}</div>
              <div class="muted">${w} ${w === 1 ? 'טבלה שלמה' : 'טבלאות שלמות'} ועוד ${p} ${p === 1 ? 'חתיכה' : 'חתיכות'} · בסך הכל ${t} עשיריות</div></div>
            </div>`;
          el.querySelector('.plus').disabled = blocked || t >= 40;
          el.querySelector('.plus10').disabled = blocked || t > 30;
          el.querySelector('.minus').disabled = blocked || t <= 0;
        }
        const set = v => { t = clamp(v, 0, 40); update(); mis.check(t); };
        el.querySelector('.plus').onclick = () => set(t + 1);
        el.querySelector('.minus').onclick = () => set(t - 1);
        el.querySelector('.plus10').onclick = () => set(t + 10);
        update();
      },
    },
    {
      t: 'משחק: כמה שוקולד?',
      r(el, done) {
        el.innerHTML = `<p>שמונה שאלות. אפשר לענות בקצב שלך, ואין שעון.</p><div class="qhost"></div>`;
        const kinds = shuffle(['pic', 'pic', 'choose', 'choose', 'words', 'count', 'half', 'words']);
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const kind = kinds[i];
            if (kind === 'pic') {
              const t = pick([ri(1, 9), ri(11, 19), ri(21, 29), ri(11, 19)]);
              return { q: 'כמה שוקולד יש כאן? כתוב את זה כמספר עשרוני.', type: 'input', answer: N.of(t, 1),
                fig: f => f.appendChild(barsSVG(t, { s: 100 })),
                hint: 'ספור קודם כמה טבלאות שלמות יש. זה המספר שלפני הנקודה. אחר כך ספור חתיכות בטבלה האחרונה.',
                explain: `${Math.floor(t / 10)} טבלאות שלמות ועוד ${t % 10} חתיכות (עשיריות): ${M(tenthStr(t))}` };
            }
            if (kind === 'choose') {
              const w = pick([0, 1]), p = ri(2, 4), t = w * 10 + p;
              const alts = w === 0 ? [t, p * 10, 10 + p] : [t, p * 10 + w, p];
              const opts = shuffle(alts).map(a => ({ h: barsSVG(a, { s: 44, min: 1 }).outerHTML, v: a }));
              return { q: `באיזו תמונה יש בדיוק ${M(tenthStr(t))} טבלאות שוקולד?`, type: 'choice', options: opts, answer: t, optCls: 'pic',
                hint: `ה-${M(String(w))} לפני הנקודה אומר כמה טבלאות שלמות, וה-${M(String(p))} אחרי הנקודה אומר כמה חתיכות.`,
                explain: `${M(tenthStr(t))} = ${w === 0 ? 'אין טבלה שלמה' : 'טבלה שלמה אחת'}, ועוד ${p} חתיכות.` };
            }
            if (kind === 'words') {
              const w = ri(1, 4), p = ri(1, 9), t = w * 10 + p;
              return { q: `כתוב כמספר עשרוני: <b>${HEB.read(N.of(t, 1))}</b>`, type: 'input', answer: N.of(t, 1),
                hint: 'השלמים באים לפני הנקודה, והעשיריות אחריה.', explain: `${M(tenthStr(t))}` };
            }
            if (kind === 'count') {
              const t = ri(11, 39);
              return { q: `כמה חתיכות (עשיריות) יש בסך הכל ב-${M(tenthStr(t))} טבלאות?`, type: 'input', answer: N.of(t), unit: 'עשיריות',
                hint: 'בכל טבלה שלמה יש 10 חתיכות.', explain: `${Math.floor(t / 10)} טבלאות הן ${Math.floor(t / 10) * 10} חתיכות, ועוד ${t % 10}: בסך הכל ${t} עשיריות.`,
                fig: f => f.appendChild(barsSVG(t, { s: 70 })) };
            }
            return { q: 'אכלת בדיוק חצי טבלה. כמה זה כמספר עשרוני?', type: 'input', answer: N.of(5, 1),
              hint: 'לכמה חתיכות מתוך 10 שווה חצי?', explain: `חצי מ-10 חתיכות זה 5 חתיכות, כלומר ${M('0.5')}.`, fig: f => f.appendChild(barsSVG(5, { s: 100 })) };
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li><b>עשירית</b> היא חלק אחד מתוך 10 חלקים שווים של שלם. ${M(`${M('0.1')} = ${F(1, 10)}`)}.</li>
          <li>הספרה מימין לנקודה סופרת עשיריות: ${M('0.7')} = שבע עשיריות = ${F(7, 10)}.</li>
          <li>הספרות משמאל לנקודה סופרות שלמים: ${M('2.4')} = שני שלמים וארבע עשיריות.</li>
          <li>10 עשיריות = שלם אחד. לכן אחרי ${M('0.9')} בא ${M('1')}, ולא "${M('0.10')}".</li>
          <li>אפשר לספור הכל בעשיריות: ${M('1.6')} = 16 עשיריות.</li>
        </ul></div>
        <p>בפרק הבא נחתוך כל חתיכה לעוד 10 חלקים קטנים. תנחש כמה חלקים כאלה יהיו בטבלה שלמה?</p>`;
      },
    },
  ],
});
