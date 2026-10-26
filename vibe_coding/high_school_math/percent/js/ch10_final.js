'use strict';
/* פרק 10: אתגר הגמר. 12 שאלות אקראיות מכל הפרקים, ואז "אז ועכשיו" מול חידות הפתיחה */

const FINAL_GEN = [
  () => { const p = pick([15, 35, 45, 12.5, 120]), x = pick([80, 240, 360, 640]); return numQ(`כמה זה ${P(p)} מ-${M(String(x))}?`, p * x / 100, { explain: `${M(`${fmt(p / 100, 3)} × ${x} = ${fmt(p * x / 100)}`)}` }); },
  () => { const [a, b] = pick([[18, 24], [7, 20], [21, 28], [9, 12], [3, 40]]); return numQ(`עומר פתר ${a} מתוך ${b} חידות בתחרות. כמה אחוז?`, 100 * a / b, { suffix: '%', unit: '%', tol: .051, explain: `${M(`${a} ÷ ${b} = ${fmt(a / b, 4)}`)} = ${P(100 * a / b, 1)}` }); },
  () => { const [p, v] = pick([[15, 12], [35, 21], [60, 45], [8, 6]]); return numQ(`${P(p)} מהתלמידים בשכבה שיחקו בטורניר, וזה ${v} תלמידים. כמה תלמידים בשכבה?`, v * 100 / p, { explain: `${M(`${v} ÷ ${fmt(p / 100)} = ${fmt(v * 100 / p)}`)}` }); },
  () => { const [c, x] = pick([[-35, 120], [15, 260], [-8, 450], [140, 25]]); return numQ(`מחיר של ${ILS(x)} ${c > 0 ? 'עלה' : 'ירד'} ב-${P(Math.abs(c))}. מה המחיר החדש?`, x * (1 + c / 100), { unit: '₪', explain: `${M(`${x} × ${fmt(1 + c / 100)} = ${fmt(x * (1 + c / 100))}`)}` }); },
  () => { const [a, b] = pick([[-30, -20], [20, -20], [50, 50], [-10, -10]]), m = (1 + a / 100) * (1 + b / 100); return numQ(`${a > 0 ? 'עלייה' : 'ירידה'} של ${P(Math.abs(a))} ואחריה ${b > 0 ? 'עלייה' : 'ירידה'} של ${P(Math.abs(b))}. מה השינוי הכולל? (ירידה: שלילי)`, (m - 1) * 100, { suffix: '%', unit: '%', explain: `${M(`${fmt(1 + a / 100)} × ${fmt(1 + b / 100)} = ${fmt(m, 4)}`)} ← ${M(pctTxt((m - 1) * 100))}` }); },
  () => { const [c, v] = pick([[-20, 96], [25, 90], [-60, 34], [10, 121]]), o = v / (1 + c / 100); return numQ(`אחרי ${c > 0 ? 'עלייה' : 'הנחה'} של ${P(Math.abs(c))} המחיר ${ILS(v)}. מה היה המחיר לפני?`, o, { unit: '₪', explain: `${M(`${v} ÷ ${fmt(1 + c / 100)} = ${fmt(o)}`)}` }); },
  () => { const x = pick([20, 25, 40, 60, 75, 80]), need = 100 * x / (100 - x); return numQ(`מניה ירדה ב-${P(x)}. בכמה אחוזים היא צריכה לעלות כדי לחזור למחיר המקורי?`, need, { suffix: '%', unit: '%', tol: .051, explain: `${M(`${F(1, fmt(1 - x / 100))} = ${fmt(1 / (1 - x / 100), 4)}`)} ← עלייה של ${P(need, 1)}` }); },
  () => { const [a, b] = pick([[4, 6], [12, 9], [25, 30], [2, 2.5]]); return numQ(`אחוז ההצבעה למפלגה ${b > a ? 'עלה' : 'ירד'} מ-${P(a)} ל-${P(b)}. בכמה <b>אחוזים</b> (יחסית) הוא השתנה? (ירידה: שלילי)`, 100 * (b - a) / a, { suffix: '%', unit: '%', tol: .051, explain: `${M(`(${fmt(b)} − ${fmt(a)}) ÷ ${fmt(a)} = ${fmt((b - a) / a, 4)}`)}. בנקודות אחוז: ${M(fmt(b - a))}.` }); },
  () => { const [a, b, v] = pick([[16, 25, 4], [48, 50, 24], [6, 50, 3], [28, 25, 7]]); return numQ(`בלי מחשבון: כמה זה ${P(a)} מ-${M(String(b))}?`, v, { explain: `${P(b)} מ-${a} = ${M(String(v))}` }); },
  () => { const r = pick([3, 6, 8, 12]); return numQ(`בערך אחרי כמה שנים מוכפל כסף בריבית דריבית של ${P(r)}?`, 72 / r, { tol: .6, explain: `חוק ה-72: ${M(`72 ÷ ${r} = ${fmt(72 / r)}`)}` }); },
  () => numQ(`אבטיח של ${M('20')} ק״ג הוא ${M('99%')} מים. אחרי ייבוש הוא ${M('96%')} מים. כמה הוא שוקל?`, 5, { unit: 'ק״ג', explain: `יבש: ${M('0.2')} ק״ג, שהם עכשיו ${M('4%')}: ${M('0.2 ÷ 0.04 = 5')} ק״ג.` }),
  () => numQ(`בכיתה א׳ 10 תלמידים ו-${M('80%')} עברו. בכיתה ב׳ 40 תלמידים ו-${M('55%')} עברו. כמה אחוז עברו בסך הכול?`, 60, { suffix: '%', unit: '%', explain: `${M('8 + 22 = 30')} מתוך ${M('50')} = ${M('60%')}` }),
];

App.add({
  id: 'ch10', short: 'אתגר הגמר', icon: '🏁', title: 'אתגר הגמר',
  desc: '12 שאלות מכל הפרקים, ומה השתנה מאז ההתחלה',
  intro: `<p>12 שאלות, אחת מכל רעיון גדול. בלי רמזים מהפרקים. מוכן?</p>`,
  steps: [
    {
      t: '12 שאלות',
      r(el, done) {
        el.innerHTML = `<div class="qh"></div><p class="muted small best"></p>`;
        const showBest = () => { el.querySelector('.best').textContent = `השיא שלך: ${Store.get('best', 0)} מתוך 12`; };
        const order = shuffle(FINAL_GEN.map((g, k) => k));
        Quiz(el.querySelector('.qh'), { rounds: 12, gen: i => FINAL_GEN[order[i]](), onDone: s => { Store.set('best', Math.max(Store.get('best', 0), s)); showBest(); done(); } });
        showBest();
      },
    },
    {
      t: 'אז ועכשיו',
      r(el, done) {
        el.innerHTML = `<p>זוכר את ארבע החידות מההתחלה? תענה עליהן שוב, ואז נשווה.</p><div class="post"></div>`;
        const box = el.querySelector('.post'), pre = Store.get('pre', {}) || {}, ans = {};
        box.innerHTML = PRETEST.map((t, k) => `<div class="pre-q" data-k="${k}"><div class="qline">${k + 1}. ${t.q}</div>
          ${t.type === 'choice' ? `<div class="opts">${t.options.map(o => `<button class="opt" data-v="${esc(o)}">${o}</button>`).join('')}</div>`
            : `<div class="inrow"><input class="num-in" dir="ltr" inputmode="decimal">${t.unit ? `<span class="unit">${t.unit}</span>` : ''}</div>`}<div class="cmp"></div></div>`).join('')
          + `<div class="controls"><button class="btn primary chk">השווה</button><span class="muted small pm"></span></div>`;
        box.querySelectorAll('.pre-q').forEach(qd => {
          const t = PRETEST[+qd.dataset.k];
          qd.querySelectorAll('.opt').forEach(b => b.onclick = () => { qd.querySelectorAll('.opt').forEach(x => x.classList.remove('on')); b.classList.add('on'); ans[t.id] = b.dataset.v; });
          const inp = qd.querySelector('input'); if (inp) inp.oninput = () => { ans[t.id] = inp.value.trim(); };
        });
        box.querySelector('.chk').onclick = () => {
          if (PRETEST.some(t => !ans[t.id])) { box.querySelector('.pm').textContent = 'תענה על כל הארבע.'; return; }
          Store.set('post', { ans, when: Date.now() });
          box.querySelectorAll('.pre-q').forEach(qd => {
            const t = PRETEST[+qd.dataset.k], a = (pre.ans || {})[t.id], b = ans[t.id], ca = pretestCorrect(t, a), cb = pretestCorrect(t, b);
            qd.querySelector('.cmp').innerHTML = `<div class="panel small">בהתחלה: <b>${a ? esc(a) : '—'}</b> ${ca === null ? '' : ca ? '✅' : '❌'} · עכשיו: <b>${esc(b)}</b> ${cb ? '✅' : '❌'}<div class="why">${t.why}</div></div>`;
          });
          const nb = PRETEST.filter(t => pretestCorrect(t, ans[t.id])).length;
          if (nb === 4) confetti();
          box.querySelector('.pm').textContent = `${nb} מתוך 4 נכונות עכשיו.`;
          done();
        };
      },
    },
  ],
});
