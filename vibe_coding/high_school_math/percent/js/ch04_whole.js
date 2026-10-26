'use strict';
/* פרק 4: מוצאים את השלם. פס בלוקים, שאלות הפוכות, והמלכודת "להוסיף את האחוז בחזרה" */

// פס בלוקים: השלם מחולק ל-100/g בלוקים, p אחוזים מהם צבועים ומסומנים v
function blockBar(p, v, opt = {}) {
  // גודל בלוק: המחלק המשותף של p ו-100, גם כש-p הוא 12.5
  const g = gcd(Math.round(p * 10), 1000) / 10, nb = 100 / g, kb = p / g, nt = Math.max(nb, kb);
  const W = 640, L = 10, w = (W - 2 * L) / nt;
  const svg = S.svg(W, 110, 'board');
  for (let k = 0; k < nt; k++) S.el('rect', { x: L + k * w + 1, y: 30, width: w - 2, height: 40, rx: Math.min(5, w / 4), class: k < kb ? 'bb-on' : 'bb-off' }, svg);
  if (opt.each !== undefined && w > 34) for (let k = 0; k < nt; k++) S.text(svg, L + k * w + w / 2, 56, fmt(opt.each), { class: 'bb-each', 'text-anchor': 'middle' });
  S.el('path', { d: `M${L},24 v-6 H${L + kb * w} v6`, class: 'bb-brace' }, svg);
  S.text(svg, clamp(L + kb * w / 2, 70, W - 70), 14, `${fmt(p)}% = ${fmt(v)}`, { class: 'bb-lbl', 'text-anchor': 'middle' });
  S.el('path', { d: `M${L},76 v6 H${L + nb * w} v-6`, class: 'bb-brace' }, svg);
  S.text(svg, clamp(L + nb * w / 2, 70, W - 70), 102, opt.whole !== undefined ? `100% = ${fmt(opt.whole)}` : '100% = ?', { class: 'bb-lbl', 'text-anchor': 'middle' });
  return svg;
}

App.add({
  id: 'ch4', short: 'מוצאים את השלם', icon: '🔍', title: 'מוצאים את השלם',
  desc: 'יודעים את החלק, מחפשים את הכול',
  intro: `<p>"${M('30%')} מהכיתה של עומר הלכו לחוג רובוטיקה. זה 9 ילדים. כמה ילדים בכיתה?" הפעם יודעים את החלק ואת האחוז, ומחפשים את <b>השלם</b>. זה הכיוון שהכי הרבה אנשים מתבלבלים בו.</p>`,
  steps: [
    {
      t: 'פס הבלוקים',
      r(el, done) {
        el.innerHTML = `<p>הרעיון: מחלקים את השלם לבלוקים שווים. אם ${M('30%')} הם 9, אז כל בלוק של ${M('10%')} הוא ${M('9 ÷ 3 = 3')}, ובשלם יש 10 בלוקים: ${M('30')} ילדים.</p>
          <p class="do">שלושה פסים. בכל פס מצא כמה שווה בלוק אחד, ואז את השלם.</p><div class="bq"></div>
          <div class="after" hidden><div class="aha"><b>הדרך הכללית:</b> מחלקים באחוז כדי להגיע ל-${M('1%')}, וכופלים ב-100. ${M('30% = 9')} ← ${M('1% = 0.3')} ← ${M('100% = 30')}.
          <br>או בקיצור: <b>${M('שלם = חלק ÷ (p / 100)')}</b>. ${M('9 ÷ 0.3 = 30')}.</div></div>`;
        const rounds = [[30, 9], [25, 14], [40, 26]];
        let r = 0;
        const box = el.querySelector('.bq');
        function draw(stage = 0) {
          if (r >= rounds.length) { box.innerHTML = '<p><b>🏆 שלושה פסים פוענחו!</b></p>'; el.querySelector('.after').hidden = false; done(); return; }
          const [p, v] = rounds[r], g = gcd(p, 100), each = v / (p / g), whole = each * 100 / g;
          box.innerHTML = `<div class="panel"><p style="font-size:1.1rem">פס ${r + 1}: ${P(p)} הם ${M(String(v))}</p><div class="fig"></div>
            <div class="inrow"><span>בלוק אחד (${P(g)}) שווה:</span><input class="num-in e1" dir="ltr" inputmode="decimal" ${stage ? `value="${fmt(each)}" disabled` : ''}></div>
            ${stage ? `<div class="inrow" style="margin-top:6px"><span>השלם (${M('100%')}) שווה:</span><input class="num-in e2" dir="ltr" inputmode="decimal"></div>` : ''}
            <div class="fb2"></div></div>`;
          box.querySelector('.fig').appendChild(blockBar(p, v, stage ? { each } : {}));
          const inp = box.querySelector(stage ? '.e2' : '.e1'), fb = box.querySelector('.fb2');
          inp.focus({ preventScroll: true });
          inp.onkeydown = e => {
            if (e.key !== 'Enter') return;
            const x = num(inp.value), want = stage ? whole : each;
            if (close(x, want)) {
              if (!stage) draw(1);
              else { inp.classList.add('right'); inp.disabled = true; box.querySelector('.fig').replaceChildren(blockBar(p, v, { each, whole })); fb.innerHTML = `<div class="aha flash">נכון! ${M(`${100 / g} × ${fmt(each)} = ${fmt(whole)}`)} <button class="btn primary sm nx">הלאה ←</button></div>`; fb.querySelector('.nx').onclick = () => { r++; draw(); }; }
            } else { inp.classList.add('wrong'); fb.innerHTML = `<div class="trap">${stage ? `בשלם יש ${100 / g} בלוקים.` : `${M(String(v))} מתחלקים בין ${p / g} הבלוקים הצבועים.`} (Enter לבדיקה)</div>`; }
          };
          fb.innerHTML = '<span class="muted small">כתוב ולחץ Enter</span>';
        }
        draw();
      },
    },
    {
      t: 'שאלות הפוכות',
      r(el, done) {
        el.innerHTML = `<p class="do">8 שאלות. כדאי לצייר פס בראש.</p><div class="qh"></div>`;
        const items = [
          [20, 7, w => `${M('20%')} מהקלפים של עומר הם נדירים, וזה 7 קלפים. כמה קלפים יש לו?`],
          [75, 36, w => `עומר קרא ${M('75%')} מהספר, כלומר 36 עמודים. כמה עמודים בספר?`],
          [40, 18, w => `${M('40%')} מהשחקנים בטורניר השחמט הם מכיתות ח׳. יש 18 שחקנים מכיתות ח׳. כמה שחקנים בטורניר?`],
          [12.5, 4, w => `${M('12.5%')} מהגרביים במגירה של עומר הן בלי זוג. 4 גרביים בלי זוג. כמה גרביים במגירה?`],
          [150, 60, w => `השנה נרשמו לחוג 60 ילדים, שזה ${M('150%')} ממספר הנרשמים בשנה שעברה. כמה נרשמו בשנה שעברה?`],
          [80, 80, w => `אחרי הנחה של ${M('20%')} חולצה עולה ${ILS(80)}. כמה עלתה לפני ההנחה?`],
          [60, 27, w => `סוללת הטלפון של עומר על ${M('60%')}, וזה מספיק ל-27 שעות המתנה. כמה שעות המתנה בסוללה מלאה?`],
          [35, 14, w => `${M('35%')} מהדרך לבית הספר עומר עובר בתוך פארק, וזה 14 דקות הליכה. כמה דקות כל הדרך?`],
          [2, 5, w => `ב-${M('2%')} מהסיבובים במשחק עומר מקבל פריט נדיר. השבוע קיבל 5 פריטים נדירים. בכמה סיבובים שיחק, אם הכול הלך לפי הסטטיסטיקה?`],
        ];
        const pool = shuffle(items);
        Quiz(el.querySelector('.qh'), {
          rounds: 8,
          gen(i) {
            const [p, v, f] = pool[i], w = v * 100 / p;
            return numQ(f(), w, { hint: `${P(p)} = ${M(String(v))}. כמה זה ${M('1%')}? ואז ${M('100%')}?`, explain: `${M('1%')} = ${M(`${v} ÷ ${fmt(p)} = ${fmt(v / p, 3)}`)}, ולכן ${M('100%')} = ${M(fmt(w))}.${p === 80 ? ' (חשוב: ה-' + M('20%') + ' הם מהמחיר המקורי, אז 80 הם ' + M('80%') + ' ממנו.)' : ''}`, fig: h => h.appendChild(blockBar(p, v)) });
          },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'המלכודת',
      r(el, done) {
        el.innerHTML = `<p>"מחיר הנעליים אחרי הנחה של ${M('20%')} הוא ${ILS(200)}. מה היה המחיר המקורי?"</p>
          <p>הרבה אנשים חושבים: "פשוט אחזיר את ה-${M('20%')}": ${M('200 + 20% = 240')}. נבדוק:</p>
          <div class="panel"><div class="controls"><span>המחיר המקורי לבדיקה:</span><input type="range" min="200" max="300" step="5" value="220" class="tr"><b class="trv"></b></div><div class="tro"></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>אחוז תמיד מחושב מהשלם שלו.</b> ההנחה היא ${M('20%')} מהמחיר <u>המקורי</u>, אז כדי לחזור צריך להוסיף ${M('20%')} ממספר גדול יותר, וזה יותר מ-${M('20%')} מ-200. ${M('200')} הם ${M('80%')} מהמחיר, ולכן המחיר ${M('200 ÷ 0.8 = 250')}.
          <br>מסקנה מפתיעה: כדי לבטל ירידה של ${M('20%')} צריך עלייה של ${M('25%')}. נבין את זה לעומק בפרק הבא.</div></div>`;
        let mis;
        const tr = el.querySelector('.tr');
        function draw() {
          const o = +tr.value, after = o * 0.8;
          el.querySelector('.trv').innerHTML = ILS(o);
          el.querySelector('.tro').innerHTML = `${ILS(o)} פחות ${M('20%')} (${ILS(o * .2)}) = <b>${ILS(after)}</b> ${after === 200 ? '✅ יוצא בדיוק 200!' : after < 200 ? '← יוצא פחות מ-200' : '← יוצא יותר מ-200'}`;
          mis && mis.check(o);
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `בדוק את ${ILS(240)}: יוצא 200?`, ok: o => o === 240 },
          { t: 'מצא את המחיר המקורי האמיתי', ok: o => o === 250 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        tr.oninput = draw; draw();
      },
    },
  ],
});
