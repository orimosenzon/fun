'use strict';
/* פרק 8: פרדוקסים. האבטיח המתייבש, פרדוקס סימפסון בכדורסל, וממוצע של אחוזים */

App.add({
  id: 'ch8', short: 'פרדוקסים', icon: '🌀', title: 'פרדוקסים של אחוזים',
  desc: 'אבטיח שמאבד חצי ממשקלו, ושחקן שטוב יותר בהכול וגרוע יותר בסך הכול',
  intro: `<p>שלוש תופעות שנראות בלתי אפשריות, ובכל אחת מהן החשבון נכון לגמרי. כל אחת מלמדת משהו עמוק על השאלה "אחוז ממה?".</p>`,
  steps: [
    {
      t: 'האבטיח המתייבש',
      r(el, done) {
        el.innerHTML = `<p>אבטיח שוקל ${M('10')} ק״ג, ו-${M('99%')} ממנו מים. הוא נשאר בשמש, וחלק מהמים התאדו. עכשיו ${M('98%')} ממנו מים. כמה הוא שוקל?</p>
          <p class="muted">רוב האנשים עונים "בערך ${M('9.9')} ק״ג". אחוז אחד של מים ירד, מה כבר יכול לקרות?</p>
          <p class="do">הזז את אחוז המים.</p>
          <div class="controls"><label class="sl wide"><span>מים</span><input type="range" min="90" max="99.5" step="0.5" value="99" class="ww"><output></output></label></div>
          <div class="play"><svg class="board mel" viewBox="0 0 520 170"></svg><div class="panel mo"></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>הסוד: החלק היבש לא משתנה.</b> בהתחלה הוא ${M('1%')} מ-10 ק״ג, כלומר ${M('0.1')} ק״ג. אחרי ההתייבשות אותם ${M('0.1')} ק״ג הם ${M('2%')} מהמשקל, ולכן המשקל הוא ${M('0.1 ÷ 0.02 = 5')} ק״ג.
          <br>ירידה מ-${M('99%')} ל-${M('98%')} מים נראית קטנה, אבל החלק ה<u>יבש</u> קפץ מ-${M('1%')} ל-${M('2%')}: הוכפל! כשמסתכלים על הצד השני של האחוז, הכול ברור.</div></div>`;
        const ww = el.querySelector('.ww'), svg = el.querySelector('.mel');
        let mis;
        function draw() {
          const w = +ww.value, tot = 0.1 / (1 - w / 100), water = tot - 0.1;
          ww.nextElementSibling.textContent = fmt(w, 1) + '%';
          S.clear(svg);
          const L = 30, R = 490, sc = (R - L) / 20, wW = water * sc, wD = Math.max(0.1 * sc, 3);
          S.el('rect', { x: L, y: 50, width: wW, height: 60, rx: 6, class: 'mel-water' }, svg);
          S.el('rect', { x: L + wW, y: 50, width: wD, height: 60, rx: 2, class: 'mel-dry' }, svg);
          S.text(svg, L, 36, `${fmt(tot, 2)} kg`, { class: 'pb-read' });
          for (let k = 0; k <= 20; k += 5) { S.el('line', { x1: L + k * sc, x2: L + k * sc, y1: 116, y2: 124, class: 'pb-tick' }, svg); S.text(svg, L + k * sc, 142, k + ' kg', { class: 'pb-lbl', 'text-anchor': 'middle' }); }
          el.querySelector('.mo').innerHTML = `<p>💧 מים: <b>${fmt(water, 2)}</b> ק״ג</p><p>🟫 יבש: <b>${M('0.1')}</b> ק״ג (${P(100 - w, 1)})</p><p style="font-size:1.3rem">סך הכול: <b>${fmt(tot, 2)}</b> ק״ג</p>`;
          mis && mis.check(w);
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `${M('98%')} מים: כמה שוקל האבטיח?`, ok: w => w === 98 },
          { t: `מתי האבטיח שוקל ${M('2')} ק״ג?`, ok: w => w === 95 },
          { t: `ומה אם מוסיפים מים עד ${M('99.5%')}?`, ok: w => w === 99.5 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        ww.oninput = draw; draw();
      },
    },
    {
      t: 'פרדוקס סימפסון',
      r(el, done) {
        el.innerHTML = `<p>המאמן צריך לבחור קלע אחד לגמר: עומר או נועם. הוא הסתכל על הזריקות שלהם בעונה:</p>
          <table class="clothes simp"><tr><th></th><th>מקרוב</th><th>מרחוק</th><th>סך הכול</th></tr>
            <tr><td><b>עומר</b></td><td>${M('9 / 10')}</td><td>${M('20 / 90')}</td><td class="tc" hidden>${M('29 / 100')}</td></tr>
            <tr><td><b>נועם</b></td><td>${M('72 / 90')}</td><td>${M('1 / 10')}</td><td class="tc" hidden>${M('73 / 100')}</td></tr></table>
          <div class="sq"></div>`;
        const qs = [
          { q: 'מי קולע טוב יותר מקרוב?', a: 'עומר', why: `עומר ${M('90%')}, נועם ${M('80%')}.` },
          { q: 'ומי קולע טוב יותר מרחוק?', a: 'עומר', why: `עומר ${M('22.2%')}, נועם ${M('10%')}.` },
          { q: 'אז מי קולע טוב יותר בסך הכול?', a: 'נועם', why: `נועם ${M('73%')}, עומר ${M('29%')}. כן, באמת! תבדוק בטבלה.`, after: () => el.querySelectorAll('.tc').forEach(t => t.hidden = false) },
        ];
        let k = 0;
        const box = el.querySelector('.sq');
        function ask() {
          if (k >= qs.length) return finale();
          const q = qs[k];
          box.insertAdjacentHTML('beforeend', `<div class="panel sqq" data-k="${k}"><p><b>${q.q}</b></p><div class="opts"><button class="opt">עומר</button><button class="opt">נועם</button></div><div class="sfb"></div></div>`);
          const p = box.lastElementChild;
          p.querySelectorAll('.opt').forEach(b => b.onclick = () => {
            p.querySelectorAll('.opt').forEach(x => x.disabled = true);
            const ok = b.textContent === q.a; b.classList.add(ok ? 'right' : 'wrong');
            if (!ok) [...p.querySelectorAll('.opt')].find(x => x.textContent === q.a).classList.add('right');
            p.querySelector('.sfb').innerHTML = `<div class="${ok ? 'aha' : 'trap'}">${q.why}</div>`;
            q.after && q.after(); k++; ask();
          });
        }
        function finale() {
          box.insertAdjacentHTML('beforeend', `<div class="aha"><b>עומר טוב יותר בכל סוג של זריקה, ובכל זאת גרוע יותר בסך הכול.</b> איך? עומר זרק כמעט רק מרחוק (90 מתוך 100), ושם קשה לקלוע. נועם זרק כמעט רק מקרוב, ושם קל. האחוז הכולל הוא בעצם <u>ממוצע משוקלל</u>, והמשקלות שלהם שונים לגמרי.</div>
            <p class="do">נניח שעומר שומר על האחוזים שלו (${M('90%')} מקרוב, ${M('22.2%')} מרחוק), אבל משנה את התמהיל. כמה מ-100 הזריקות שלו צריכות להיות מקרוב כדי לעקוף את נועם בסך הכול?</p>
            <div class="controls"><label class="sl wide"><span>מקרוב</span><input type="range" min="0" max="100" step="1" value="10" class="mix"><output></output></label></div>
            <div class="panel mixo"></div><div class="mh"></div>
            <div class="tip">פרדוקס סימפסון קורה באמת: בניסויים רפואיים, בקבלה לאוניברסיטאות ובספורט. כשמשווים אחוזים כוללים, תמיד שואלים: <b>האם התמהיל זהה?</b></div>`);
          const mix = box.querySelector('.mix');
          const mis = Missions(box.querySelector('.mh'), [{ t: 'מצא את המספר הקטן ביותר של זריקות מקרוב שבו עומר עוקף את נועם', ok: c => c === 75 }], () => done());
          const draw = () => { const c = +mix.value, t = .9 * c + (2 / 9) * (100 - c); mix.nextElementSibling.textContent = c; box.querySelector('.mixo').innerHTML = `עומר בסך הכול: <b>${P(t, 1)}</b> ${t > 73 ? '✅ יותר מנועם' : `(נועם: ${M('73%')})`}`; mis.check(c); };
          mix.oninput = draw; draw();
        }
        ask();
      },
    },
    {
      t: 'ממוצע של אחוזים',
      r(el, done) {
        el.innerHTML = `<p>עומר קיבל ${M('100%')} בבוחן של 2 שאלות, ו-${M('50%')} במבחן של 40 שאלות. הוא אומר: "הממוצע שלי ${M('75%')}". נכון?</p>
          <div class="panel">בסך הכול הוא ענה נכון על ${M('2 + 20 = 22')} שאלות מתוך ${M('42')}, כלומר ${P(2200 / 42, 1)}. הבוחן הקטן כמעט לא משפיע.</div>
          <div class="def">אי אפשר סתם לעשות ממוצע לאחוזים שמחושבים <b>משלמים שונים</b>. צריך לחזור לכמויות: לחבר את החלקים, לחבר את השלמים, ורק אז לחלק.</div>
          <p class="do">4 שאלות.</p><div class="qh"></div>`;
        const qs = [
          numQ(`בכיתה ח׳1 יש 20 תלמידים ו-${M('90%')} מהם עברו את המבחן. בח׳2 יש 30 תלמידים ו-${M('60%')} עברו. כמה אחוז עברו בשתי הכיתות יחד?`, 72, { suffix: '%', unit: '%', hint: 'כמה תלמידים עברו בכל כיתה?', explain: `${M('18 + 18 = 36')} מתוך ${M('50')}, כלומר ${M('72%')} (לא ${M('75%')}).` }),
          numQ(`עומר נסע 60 ק״מ במהירות 60 קמ״ש, וחזר באותה דרך במהירות 40 קמ״ש. מה המהירות הממוצעת שלו?`, 48, { unit: 'קמ״ש', hint: 'כמה זמן לקח כל כיוון? מהירות ממוצעת = כל הדרך ÷ כל הזמן.', explain: `הלוך: שעה. חזור: ${M('1.5')} שעות. ${M('120 ÷ 2.5 = 48')} קמ״ש, לא 50! (אותו רעיון: לא מחשבים ממוצע של יחסים בלי לשקלל.)` }),
          { q: `בשבוע הראשון עומר ניצח ב-${M('80%')} מ-5 משחקים, ובשבוע השני ב-${M('40%')} מ-20 משחקים. הממוצע הכולל שלו:`, type: 'choice', options: [{ h: `בדיוק ${M('60%')}`, v: 'a' }, { h: `פחות מ-${M('60%')}`, v: 'b' }, { h: `יותר מ-${M('60%')}`, v: 'c' }], answer: 'b', hint: 'באיזה שבוע היו יותר משחקים?', explain: `${M('4 + 8 = 12')} ניצחונות מתוך 25, כלומר ${M('48%')}. השבוע עם הרבה משחקים "מושך" את הממוצע אליו.` },
          { q: `מתי כן מותר לחשב ממוצע רגיל של שני אחוזים?`, type: 'choice', options: [{ h: 'כשהשלמים שווים בגודל', v: 'a' }, { h: 'כשהאחוזים קרובים', v: 'b' }, { h: 'אף פעם', v: 'c' }], answer: 'a', explain: `אם בשתי הכיתות יש 25 תלמידים, אז ${M('(90% + 60%) ÷ 2 = 75%')} באמת נכון.` },
        ];
        Quiz(el.querySelector('.qh'), { rounds: 4, gen: i => qs[i], onDone: () => done() });
      },
    },
  ],
});
