'use strict';
/* פרק 2: אחוז מכמות. פס כפול, אבני בניין (50%, 10%, 5%, 1%), הטריק של ההחלפה, ותרגול בראש */

App.add({
  id: 'ch2', short: 'אחוז מכמות', icon: '🧮', title: 'כמה זה 35% מ-80?',
  desc: 'פס כפול, אבני בניין, והטריק של ההחלפה',
  intro: `<p>שלוש דרכים לחשב אחוז מכמות. כל אחת מראה משהו אחר, ועם שלושתן אפשר לחשב כמעט הכול בראש.</p>`,
  steps: [
    {
      t: 'הפס הכפול',
      r(el, done) {
        el.innerHTML = `<p>ציור שמתמטיקאים אוהבים: פס אחד, ושני סרגלים עליו. למעלה אחוזים, למטה הכמות. ${M('0%')} תמיד מול 0, ו-${M('100%')} תמיד מול השלם.</p>
          <p class="do">עומר חוסך לאופניים שעולים ${ILS(1200)}. גרור את הידית.</p>
          <div class="bh"></div><div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>החשבון שמאחורי הפס:</b> ${M('1%')} הוא החלק המאה, ${M('1200 ÷ 100 = 12')}. מכאן כל אחוז הוא פשוט כפל: ${M('35% = 35 × 12 = 420')}.
          <br>בקיצור: <b>${M('p%')} מ-${M('x')} = ${M('p ÷ 100 × x')}</b>. ובשפה של שברים עשרוניים: ${M('35%')} מ-1200 זה ${M('0.35 × 1200')}.</div></div>`;
        let mis;
        PctBar(el.querySelector('.bh'), { whole: 1200, step: 1, unit: '₪', pct: 0, onChange: (p, v) => mis && mis.check({ p, v }) });
        mis = Missions(el.querySelector('.mh'), [
          { t: `חצי מהמחיר: כמה אחוז זה?`, ok: s => s.p === 50 },
          { t: `${M('10%')}: כמה שקלים?`, ok: s => s.p === 10 },
          { t: `${M('1%')}: כמה שקלים? (צריך יד יציבה)`, ok: s => s.p === 1 },
          { t: `עומר כבר חסך ${ILS(420)}. כמה אחוז מהמחיר זה?`, ok: s => s.v === 420 },
          { t: `${M('75%')}: כמה עוד חסר לו מכאן?`, ok: s => s.p === 75, after: () => note(`חסרים עוד ${M('25%')}, כלומר ${ILS(300)}.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const x = document.createElement('div'); x.className = 'tip flash'; x.innerHTML = h; el.querySelector('.mh').appendChild(x); }
      },
    },
    {
      t: 'אבני בניין',
      r(el, done) {
        el.innerHTML = `<p>את ${M('10%')} קל לחשב: מזיזים את הנקודה מקום אחד. ${M('10%')} מ-260 זה 26. חצי מזה (${M('5%')}) זה 13. ו-${M('1%')} זה 2.6. מאבני הבניין האלה אפשר <b>להרכיב כל אחוז</b>.</p>
          <p class="do">לחץ על האבנים כדי להרכיב את האחוז המבוקש. אפשר גם להוריד (${M('−')}). 4 סבבים.</p>
          <div class="panel bq"></div>
          <div class="after" hidden><div class="aha"><b>זה בדיוק מה שאנשים עושים בראש.</b> ${M('15%')} טיפ במסעדה? ${M('10% + 5%')}. ${M('19%')}? ${M('20% − 1%')}. מתמטית, זה חוק הפילוג: ${M('(10% + 5%) × x = 10% × x + 5% × x')}.</div></div>`;
        const box = el.querySelector('.bq');
        const rounds = [[35, 260], [15, 84], [19, 300], [62, 450]];
        let r = 0, used = [];
        const blocks = [100, 50, 25, 10, 5, 1];
        function draw() {
          if (r >= rounds.length) { box.innerHTML = `<p><b>🏆 בנית ארבעה אחוזים!</b></p>`; el.querySelector('.after').hidden = false; done(); return; }
          const [tp, x] = rounds[r];
          const sumP = used.reduce((a, b) => a + b, 0), sumV = used.reduce((a, b) => a + b * x / 100, 0);
          box.innerHTML = `<p style="font-size:1.15rem">סבב ${r + 1}: בנה <b>${P(tp)} מ-${M(String(x))}</b></p>
            <div class="controls">${blocks.map(b => `<button class="btn blk-btn" data-b="${b}">+ ${M(b + '%')}<small>${fmt(b * x / 100)}</small></button>`).join('')}</div>
            <div class="controls">${blocks.map(b => `<button class="btn sm sub" data-b="${b}">− ${M(b + '%')}</button>`).join('')}<button class="btn sm clr">נקה</button></div>
            <div class="stack">${used.map(b => `<span class="chip ${b < 0 ? 'neg' : ''}">${M((b < 0 ? '−' : '+') + Math.abs(b) + '%')} <small>${fmt(Math.abs(b) * x / 100)}</small></span>`).join('') || '<span class="muted">עוד לא לקחת אבנים</span>'}</div>
            <p>סך הכול: <b>${P(sumP)}</b> = <b>${M(fmt(sumV))}</b></p>
            ${sumP === tp ? `<div class="aha flash">יש! ${P(tp)} מ-${x} = ${M(fmt(sumV))} <button class="btn primary sm nx">הלאה ←</button></div>` : ''}`;
          box.querySelectorAll('.blk-btn').forEach(b => b.onclick = () => { used.push(+b.dataset.b); draw(); });
          box.querySelectorAll('.sub').forEach(b => b.onclick = () => { used.push(-b.dataset.b); draw(); });
          box.querySelector('.clr').onclick = () => { used = []; draw(); };
          const nx = box.querySelector('.nx'); if (nx) nx.onclick = () => { r++; used = []; draw(); };
        }
        draw();
      },
    },
    {
      t: 'הטריק של ההחלפה',
      r(el, done) {
        el.innerHTML = `<p>כמה זה ${M('8%')} מ-50? נראה מסובך. אבל ${M('50%')} מ-8 זה קל: 4. ו... זו אותה תשובה. <b>תמיד.</b></p>
          <div class="vs"><div class="panel"><b>למה?</b><br>${M('8%')} מ-50 = ${M(`${F(8, 100)} × 50`)} = ${M(`${F('8 × 50', 100)}`)}<br>${M('50%')} מ-8 = ${M(`${F(50, 100)} × 8`)} = ${M(`${F('50 × 8', 100)}`)}<br>ובכפל הסדר לא משנה.</div>
          <div class="panel"><b>בציור:</b> מלבן של ${M('a × b')} ריבועים, מחולק ב-100. לא משנה אם מסתכלים עליו לאורך או לרוחב.<svg class="board swap" viewBox="0 0 260 140"></svg></div></div>
          <p class="do">תמצא בכל שאלה את הצד הקל. 6 שאלות, בלי מחשבון.</p><div class="qh"></div>`;
        const sv = el.querySelector('.swap');
        for (let i = 0; i < 8; i++) for (let j = 0; j < 4; j++) S.el('rect', { x: 20 + i * 28, y: 10 + j * 28, width: 26, height: 26, rx: 3, class: 'cell pc' }, sv);
        S.text(sv, 135, 134, '8 × 4 = 4 × 8', { 'text-anchor': 'middle', class: 'pb-lbl big' });
        const pairs = [[8, 50, 4], [4, 25, 1], [12, 50, 6], [16, 25, 4], [18, 50, 9], [7, 20, 1.4], [3, 200, 6], [44, 25, 11], [36, 50, 18], [24, 75, 18], [2, 350, 7], [32, 25, 8]];
        const pool = shuffle(pairs);
        Quiz(el.querySelector('.qh'), {
          rounds: 6,
          gen(i) { const [a, b, v] = pool[i]; return numQ(`כמה זה ${P(a)} מ-${M(String(b))}?`, v, { hint: `נסה להחליף: ${P(b)} מ-${M(String(a))}.`, explain: `${P(a)} מ-${b} = ${P(b)} מ-${a} = ${M(fmt(v))}` }); },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'תרגול בראש',
      r(el, done) {
        el.innerHTML = `<p class="do">10 שאלות. תבחר בכל שאלה את השיטה הכי נוחה: פס, אבני בניין, או החלפה.</p><div class="qh"></div>`;
        Quiz(el.querySelector('.qh'), {
          rounds: 10,
          gen() {
            const p = pick([5, 10, 15, 20, 25, 30, 40, 45, 60, 75, 80, 90, 1, 2, 12.5, 150, 110, 0.5]);
            let x; do { x = pick([20, 40, 60, 80, 120, 160, 200, 240, 300, 360, 400, 480, 500, 640, 800, 1000, 1200]); } while (!Number.isInteger(p * x / 100 * 10));
            const v = p * x / 100;
            const ctx = pick([`כמה זה ${P(p)} מ-${M(String(x))}?`, `בתחרות השתתפו ${x} ילדים, ו-${P(p)} מהם עלו לגמר. כמה עלו לגמר?`, `באוסף של עומר ${x} קלפים. ${P(p)} מהם נדירים. כמה קלפים נדירים?`]);
            return numQ(ctx, v, { hint: `התחל מ-${M('10%')} (${M(fmt(x / 10))}) או מ-${M('1%')} (${M(fmt(x / 100))}).`, explain: `${M('1%')} = ${M(fmt(x / 100))}, אז ${P(p)} = ${M(`${fmt(p)} × ${fmt(x / 100)} = ${fmt(v)}`)}` });
          },
          onDone: () => done(),
        });
      },
    },
  ],
});
