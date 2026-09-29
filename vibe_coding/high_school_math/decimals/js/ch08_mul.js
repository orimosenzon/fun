'use strict';
/* פרק 8: כפל. שלם כפול עשרוני (חיבור חוזר), עשרוני כפול עשרוני (שטח בריבוע), והכלל של ספירת הספרות. */

// מחוון קטן: תווית, טווח, ופונקציית תצוגה
function slider(label, min, max, val, fmt) {
  return `<label class="sl"><span>${label}</span><input type="range" min="${min}" max="${max}" value="${val}" step="1"><output>${fmt(val)}</output></label>`;
}

App.add({
  id: 'ch8', short: 'כפל', icon: '✖️', title: 'כפל שברים עשרוניים',
  desc: 'למה כפל יכול להקטין, ואיפה שמים את הנקודה',
  intro: `<p>עד עכשיו חשבנו שכפל תמיד מגדיל: ${M('3 × 4 = 12')}. בשברים עשרוניים מחכה לך הפתעה.</p>`,
  steps: [
    {
      t: 'שלם כפול עשרוני',
      r(el, done) {
        el.innerHTML = `
          <p>${M('3 × 0.4')} פירושו "3 פעמים ${M('0.4')}", כלומר ${M('0.4 + 0.4 + 0.4')}. בשוקולד: שלושה ילדים, ולכל אחד 4 חתיכות. כמה שוקולד יש ביחד?</p>
          <p class="do">בחר כמה פעמים וכמה בכל פעם, ואז לחץ "לאחד" כדי לחבר את כל החתיכות לטבלאות.</p>
          <div class="controls">${slider('כמה פעמים', 1, 6, 3, v => v)}${slider('כמה בכל פעם', 1, 9, 4, v => '0.' + v)}</div>
          <div class="groups" style="display:flex;flex-wrap:wrap;gap:10px;direction:ltr"></div>
          <div class="controls"><button class="btn primary join">🧲 לאחד</button></div>
          <div class="joined"></div>
          <div class="panel ro"></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha">כופלים שלם בעשרוני כמו שכופלים מספרים רגילים, <b>אבל סופרים בעשיריות</b>: ${M('3 × 4')} עשיריות = 12 עשיריות = ${M('1.2')}. ${M('5 × 0.2 = 1')}, כי 10 עשיריות הן שלם.</div></div>`;
        const [sn, sd] = el.querySelectorAll('input[type=range]');
        let joinedFor = null;
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `גלה כמה זה ${M('3 × 0.4')}`, ok: s => s === '3x4' },
          { t: `גלה כמה זה ${M('5 × 0.2')}`, ok: s => s === '5x2', after: () => note(`${M('5 × 0.2 = 1')}. בדיוק טבלה שלמה! כי 10 עשיריות הן שלם.`) },
          { t: `גלה כמה זה ${M('6 × 0.5')}`, ok: s => s === '6x5', after: () => note(`6 חצאים הם 3 שלמים: ${M('6 × 0.5 = 3')}. כפל ב-${M('0.5')} זה כמו לקחת חצי.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function draw() {
          const n = +sn.value, d = +sd.value;
          sn.nextElementSibling.textContent = n; sd.nextElementSibling.textContent = '0.' + d;
          const G = el.querySelector('.groups'); G.innerHTML = '';
          for (let k = 0; k < n; k++) { const s = barsSVG(d, { s: 70 }); s.style.maxWidth = '84px'; G.appendChild(s); }
          el.querySelector('.joined').innerHTML = '';
          el.querySelector('.ro').innerHTML = `${M(`${n} × 0.${d} = ${Array(n).fill('0.' + d).join(' + ')} = ?`)}`;
          joinedFor = null;
        }
        el.querySelector('.join').onclick = () => {
          const n = +sn.value, d = +sd.value, t = n * d;
          const J = el.querySelector('.joined'); J.innerHTML = '<p class="muted small">כל החתיכות ביחד, מסודרות בטבלאות:</p>'; J.appendChild(barsSVG(t, { s: 90 }));
          el.querySelector('.ro').innerHTML = `${M(`${n} × 0.${d}`)} = ${n} × ${d} עשיריות = <b>${t} עשיריות</b> = <span class="bignum" style="font-size:1.8rem">${digitsHTML(N.str(N.of(t, 1)))}</span>`;
          joinedFor = n + 'x' + d; mis.check(joinedFor);
        };
        sn.oninput = sd.oninput = draw;
        draw();
      },
    },
    {
      t: 'עשרוני כפול עשרוני: שטח',
      r(el, done) {
        el.innerHTML = `
          <p>ומה זה ${M('0.3 × 0.4')}? "שלוש עשיריות פעמים"? קשה לדמיין. אבל יש דרך: <b>שטח של מלבן</b>. מלבן שרוחבו 3 ואורכו 4 הוא בשטח ${M('3 × 4 = 12')}. אז מלבן שרוחבו ${M('0.3')} ואורכו ${M('0.4')} הוא בשטח ${M('0.3 × 0.4')}.</p>
          <p>הריבוע הגדול הוא 1 על 1, כלומר שטחו 1. כל ריבוע קטן הוא מאית.</p>
          <p class="do">הזז את המחוונים. הרוחב צובע עמודות בכחול, הגובה צובע שורות בכתום, והמלבן שנוצר בחפיפה (הסגול) הוא התוצאה.</p>
          <div class="play">
            <div><div class="gh"></div><label class="chk"><input type="checkbox" class="big"> ריבוע גדול יותר (2 על 2), בשביל מספרים גדולים מ-1</label></div>
            <div><div class="controls" style="flex-direction:column;align-items:flex-start">${slider('רוחב ↔', 1, 10, 3, v => N.str(N.of(v, 1)))}${slider('גובה ↕', 1, 10, 4, v => N.str(N.of(v, 1)))}</div>
              <div class="panel ro"></div><div class="mhost"></div></div>
          </div>
          <div class="after" hidden><div class="aha"><b>ההפתעה:</b> ${M('0.3 × 0.4 = 0.12')}, וזה <b>קטן</b> משני המספרים! כשכופלים במספר קטן מ-1, לוקחים רק <b>חלק</b> ממשהו. ${M('0.4')} פעמים משהו זה פחות ממנו. ${M('0.5 × 0.5')} זה חצי מחצי, כלומר רבע.</div></div>`;
        const [sa, sb] = el.querySelectorAll('input[type=range]'), bigBox = el.querySelector('.big');
        const svg = S.svg(300, 300, 'board'); svg.style.maxWidth = '360px'; el.querySelector('.gh').appendChild(svg);
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `הראה את ${M('0.3 × 0.4')}`, ok: s => s === '3,4' || s === '4,3' },
          { t: `הראה את ${M('0.5 × 0.5')}`, ok: s => s === '5,5' },
          { t: `הראה את ${M('0.1 × 0.1')}`, ok: s => s === '1,1', after: () => note(`עשירית כפול עשירית = <b>מאית</b>. ריבוע קטן אחד! ${M('0.1 × 0.1 = 0.01')}`) },
          { t: `הראה את ${M('1.2 × 1.5')} (סמן את הריבוע הגדול)`, ok: s => s === '12,15' || s === '15,12', after: () => note(`כאן התוצאה ${M('1.8')} גדולה משני המספרים, כי שניהם גדולים מ-1.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function draw() {
          const B = bigBox.checked ? 20 : 10;
          sa.max = B; sb.max = B;
          const a = +sa.value, b = +sb.value;
          sa.nextElementSibling.textContent = N.str(N.of(a, 1)); sb.nextElementSibling.textContent = N.str(N.of(b, 1));
          S.clear(svg);
          const X = 10, s = 280, w = s / B;
          S.el('rect', { x: X - 2, y: X - 2, width: s + 4, height: s + 4, rx: 5, class: 'grid-frame' }, svg);
          for (let c = 0; c < B; c++) for (let r = 0; r < B; r++) {
            const inA = c < a, inB = r < b;
            S.el('rect', { x: X + c * w + .5, y: X + s - (r + 1) * w + .5, width: w - 1, height: w - 1, class: 'cell ' + (inA && inB ? 'ab' : inA ? 'a' : inB ? 'b' : '') }, svg);
          }
          if (B === 20) { S.el('line', { x1: X + s / 2, y1: X, x2: X + s / 2, y2: X + s, stroke: '#111', 'stroke-width': 2.5 }, svg); S.el('line', { x1: X, y1: X + s / 2, x2: X + s, y2: X + s / 2, stroke: '#111', 'stroke-width': 2.5 }, svg); }
          const p = a * b, A = N.str(N.of(a, 1)), Bs = N.str(N.of(b, 1));
          el.querySelector('.ro').innerHTML = `<div>${M(`${A} × ${Bs}`)} = ${a} עמודות × ${b} שורות = <b>${p} ריבועים סגולים</b></div>
            <div>${p} מאיות = <span class="bignum" style="font-size:2rem">${digitsHTML(N.str(N.of(p, 2)))}</span></div>
            <div class="muted small">${p < a * 10 && p < b * 10 ? '🔻 התוצאה קטנה משני המספרים!' : p > a * 10 && p > b * 10 ? '🔺 התוצאה גדולה משני המספרים.' : ''}</div>`;
          mis.check(`${a},${b}`);
        }
        sa.oninput = sb.oninput = bigBox.onchange = draw;
        draw();
      },
    },
    {
      t: 'איפה שמים את הנקודה?',
      r(el, done) {
        el.innerHTML = `
          <p>לצבוע ריבועים כל פעם זה ארוך. אז בוא נבין את הקיצור. ${M('0.3')} זה ${F(3, 10)}, ו-${M('0.4')} זה ${F(4, 10)}. כופלים: ${M(`${F(3, 10)} × ${F(4, 10)} = ${F(12, 100)} = ${M('0.12')}`)}.</p>
          <div class="def"><b>הקיצור:</b>
            <ol style="margin:4px 0"><li>מתעלמים מהנקודות וכופלים כמו מספרים רגילים: ${M('3 × 4 = 12')}.</li>
            <li>סופרים כמה ספרות יש אחרי הנקודה <b>בשני המספרים ביחד</b>: ב-${M('0.3')} אחת, ב-${M('0.4')} אחת. ביחד 2.</li>
            <li>בתוצאה צריכות להיות בדיוק 2 ספרות אחרי הנקודה: ${M('0.12')}.</li></ol>
            למה זה עובד? כל ספרה אחרי הנקודה היא "חלקי 10". שתי ספרות = חלקי ${M('10 × 10 = 100')}.</div>
          <div class="trap"><b>זהירות מהאפס:</b> ${M('0.5 × 0.4')}: ${M('5 × 4 = 20')}, ושתי ספרות אחרי הנקודה: ${M('0.20')}, וזה ${M('0.2')}. קודם שמים את הנקודה, ורק אז מוחקים אפסים בסוף.</div>
          <p class="tip"><b>ובודקים עם הערכה:</b> ${M('2.4 × 0.3')}? ${M('2.4')} זה בערך 2, ו-${M('0.3')} זה קצת פחות משליש. אז התוצאה בערך ${M('0.7')}. אם יצא לך ${M('7.2')}, משהו לא בסדר.</p>
          <div class="qhost"></div>`;
        const kinds = shuffle(['dot', 'dot', 'dot', 'calc', 'calc', 'calc', 'size', 'calc']);
        const mk = () => {
          const opts = [[`0.${ri(2, 9)}`, `0.${ri(2, 9)}`], [`${ri(2, 9)}`, `0.${ri(2, 9)}`], [`${ri(1, 4)}.${ri(1, 9)}`, `0.${ri(2, 9)}`], [`0.${ri(1, 9)}${ri(1, 9)}`, `${ri(2, 5)}`], [`${ri(1, 3)}.${ri(1, 9)}`, `${ri(1, 2)}.${ri(1, 9)}`], [`0.0${ri(2, 9)}`, `0.${ri(2, 9)}`]];
          return pick(opts);
        };
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const [a, b] = mk(), va = N.parse(a), vb = N.parse(b), r = N.mul(va, vb), rs = N.str(r);
            const pa = N.places(va), pb = N.places(vb);
            const whole = String(Number(a.replace('.', '')) * Number(b.replace('.', '')));
            const why = `${M(`${a.replace('.', '').replace(/^0+/, '')} × ${b.replace('.', '').replace(/^0+/, '')} = ${whole}`)}, ו${pa + pb === 1 ? 'ספרה אחת' : pa + pb + ' ספרות'} אחרי הנקודה (${pa} ב-${M(a)} ו-${pb} ב-${M(b)}): ${M(`${a} × ${b} = ${N.str(r, pa + pb)}`)}${N.str(r, pa + pb) !== rs ? ` = ${M(rs)}` : ''}.`;
            if (kinds[i] === 'dot') {
              const cands = [0, 1, 2, 3, 4].map(k => N.str(N.of(Number(whole), k))).filter((x, j, arr) => arr.indexOf(x) === j);
              return choiceQ(`${M(`${a} × ${b}`)}: כבר חישבנו ש-${M(`${a.replace('.', '').replace(/^0+/, '')} × ${b.replace('.', '').replace(/^0+/, '')} = ${whole}`)}. איפה הנקודה?`, rs, shuffle(cands.filter(x => x !== rs)).slice(0, 3), { fmt: M, hint: 'ספור כמה ספרות יש אחרי הנקודה בשני המספרים ביחד. וגם תעריך: בערך כמה זה צריך לצאת?', explain: why });
            }
            if (kinds[i] === 'size') {
              const n = ri(6, 40), d = pick(['0.5', '0.9', '0.1', '1.5', '0.25']), bigger = N.parse(d) > U;
              return choiceQ(`בלי לחשב: ${M(`${n} × ${d}`)} גדול מ-${M(String(n))} או קטן ממנו?`, bigger ? `גדול מ-${n}` : `קטן מ-${n}`, [bigger ? `קטן מ-${n}` : `גדול מ-${n}`, `שווה ל-${n}`],
                { hint: `האם ${M(d)} גדול מ-1 או קטן מ-1?`, explain: `${M(d)} ${bigger ? 'גדול' : 'קטן'} מ-1, אז לוקחים ${bigger ? 'יותר מ' : 'רק חלק מ'}-${n}. ${M(`${n} × ${d} = ${N.str(N.mul(N.of(n), N.parse(d)))}`)}` });
            }
            return { q: `${M(`${a} × ${b} = ?`)}`, type: 'input', answer: r, hint: 'כפול בלי הנקודות, ואז ספור ספרות אחרי הנקודה.', explain: why };
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li>שלם כפול עשרוני = חיבור חוזר: ${M('3 × 0.4 = 0.4 + 0.4 + 0.4 = 1.2')}.</li>
          <li>עשרוני כפול עשרוני = שטח של מלבן בריבוע: ${M('0.3 × 0.4 = 0.12')} (12 ריבועים מתוך 100).</li>
          <li><b>כפל במספר קטן מ-1 מקטין!</b> ${M('8 × 0.5 = 4')}. כפל במספר גדול מ-1 מגדיל.</li>
          <li>הקיצור: כופלים בלי נקודות, ואז סופרים ספרות אחרי הנקודה בשני המספרים ביחד. ${M('1.2 × 0.03')}: ${M('12 × 3 = 36')}, 3 ספרות, ${M('0.036')}.</li>
          <li>קודם שמים נקודה, אחר כך מוחקים אפסים בסוף: ${M('0.5 × 0.4 = 0.20 = 0.2')}.</li>
          <li>תמיד בודקים עם הערכה.</li>
        </ul></div>`;
      },
    },
  ],
});
