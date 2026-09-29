'use strict';
/* פרק 5: מי גדול יותר? מכונת השוואה עם יישור לפי הנקודה, דו-קרב מלכודות, ומיון. */

// טבלת מקומות עם שני מספרים זה מעל זה, מיושרים לפי הנקודה, עם אפסי רפאים
function compareTable(a, b) {
  const [ai, af = ''] = a.split('.'), [bi, bf = ''] = b.split('.');
  const I = Math.max(ai.length, bi.length), Fn = Math.max(af.length, bf.length, 1);
  const cols = []; for (let p = I - 1; p >= 0; p--) cols.push(p); for (let p = 1; p <= Fn; p++) cols.push(-p);
  const dig = (ip, fp, p) => p >= 0 ? (ip.length - 1 - p >= 0 ? { d: ip[ip.length - 1 - p], g: false } : { d: '', g: true }) : (fp[-p - 1] !== undefined ? { d: fp[-p - 1], g: false } : { d: '0', g: true });
  let diff = null;
  for (const p of cols) { const x = dig(ai, af, p).d || '0', y = dig(bi, bf, p).d || '0'; if (x !== y) { diff = p; break; } }
  const row = (ip, fp) => cols.map((p, k) => `${p === -1 ? '<td class="dotc">.</td>' : ''}<td class="${PLACE_CLS[p]} ${dig(ip, fp, p).g ? 'ghost' : ''} ${p === diff ? 'diff' : ''}">${dig(ip, fp, p).d}</td>`).join('');
  const head = cols.map(p => `${p === -1 ? '<th class="dotc"></th>' : ''}<th class="${PLACE_CLS[p]}">${PLACE_NAME[p]}</th>`).join('');
  return { html: `<table class="pv"><tr>${head}</tr><tr>${row(ai, af)}</tr><tr>${row(bi, bf)}</tr></table>`, diff, last: cols[cols.length - 1] };
}
function compareWhy(a, b) {
  const va = N.parse(a), vb = N.parse(b), { html, diff, last } = compareTable(a, b);
  if (va === vb) return `${html}<p>אחרי שמשלימים אפסים, כל הספרות זהות. ${M(a + ' = ' + b)}: זה אותו מספר בדיוק.</p>`;
  const big = va > vb ? a : b, small = va > vb ? b : a;
  const getD = (s, p) => { const [i, f = ''] = s.split('.'); return p >= 0 ? (i[i.length - 1 - p] || '0') : (f[-p - 1] || '0'); };
  return `${html}<p>משווים משמאל לימין. ההבדל הראשון הוא במקום של ה<b>${PLACE_NAME[diff]}</b>: ${M(getD(big, diff))} מול ${M(getD(small, diff))}. לכן ${M(big + ' > ' + small)}. ${diff !== last ? 'מה שבא אחרי זה כבר לא משנה.' : ''}</p>`;
}
// זוגות שבנויים בדיוק על הטעויות הנפוצות
function trapPair(kind) {
  const d = ri(3, 9);
  switch (kind) {
    case 'long': { const x = ri(1, d - 1), y = ri(1, 9); return [`0.${d}`, `0.${x}${y}`]; }          // 0.8 מול 0.75
    case 'short': { const y = ri(1, 9); return [`0.${d}`, `0.${d}${y}`]; }                           // 0.4 מול 0.45
    case 'zeroEnd': return [`0.${d}`, `0.${d}0`];                                                    // 0.5 מול 0.50
    case 'zeroMid': return [`0.0${d}`, `0.${d}`];                                                    // 0.05 מול 0.5
    case 'nines': { const t = ri(1, 8); return [`0.${t - 1 < 0 ? 0 : t - 1}9`, `0.${t}`]; }            // 0.09 מול 0.1 / 0.29 מול 0.3
    case 'whole': { const w = ri(1, 4); return [`${w}.${ri(1, 4)}`, `${w - 1}.${ri(6, 9)}${ri(1, 9)}`]; } // 1.2 מול 0.95
    case 'wzero': { const w = ri(1, 5); return [`${w}.0${d}`, `${w}.${d}`]; }                         // 2.05 מול 2.5
    case 'thou': { const x = ri(1, 8); return [`0.${x}${ri(1, 9)}${ri(1, 9)}`, `0.${x + 1}`]; }        // 0.125 מול 0.2
    default: { const x = ri(2, 9); return [`0.${x}`, `0.${x - 1}99`]; }                               // 0.7 מול 0.699
  }
}

App.add({
  id: 'ch5', short: 'מי גדול?', icon: '⚖️', title: 'מי גדול יותר?',
  desc: 'המלכודת הכי נפוצה בשברים עשרוניים, ואיך לא ליפול בה',
  intro: `<p>זוכר את השאלה מההתחלה, ${M('0.8')} או ${M('0.75')}? זו המלכודת הכי נפוצה בשברים עשרוניים. הרבה מבוגרים נופלים בה. בפרק הזה תלמד שיטה שלא נכשלת אף פעם.</p>`,
  steps: [
    {
      t: 'שלוש דרכים לחשוב',
      r(el, done) {
        el.innerHTML = `
          <p>שלושה חברים מתווכחים מי גדול יותר. כל אחד עם שיטה משלו:</p>
          <div class="cards">
            <div class="sum-card"><h4>😎 נועה</h4><p>"המספר ה<b>ארוך</b> יותר גדול יותר. ${M('0.75')} גדול מ-${M('0.8')}, כי 75 גדול מ-8."</p></div>
            <div class="sum-card"><h4>🤓 תום</h4><p>"המספר ה<b>קצר</b> יותר גדול יותר, כי עשיריות גדולות ממאיות. לכן ${M('0.4')} גדול מ-${M('0.45')}."</p></div>
            <div class="sum-card"><h4>🧐 מאיה</h4><p>"משלימים אפסים עד ששני המספרים באותו אורך, ואז משווים ספרה-ספרה משמאל."</p></div>
          </div>
          <div class="modal-q">מי מהם צודק תמיד?
            <div class="opts" style="margin-top:8px"><button class="opt" data-v="n">נועה</button><button class="opt" data-v="t">תום</button><button class="opt" data-v="m">מאיה</button><button class="opt" data-v="x">אף אחד</button></div><div class="fbq"></div></div>
          <div class="reveal" hidden>
            <div class="vs">
              <div class="panel"><b>נועה טועה:</b> ${M('0.8')} = 80 מאיות, ו-${M('0.75')} = 75 מאיות. בשקלים: 80 אגורות מול 75 אגורות. <div class="g-n"></div></div>
              <div class="panel"><b>תום טועה:</b> ${M('0.4')} = 40 מאיות, ו-${M('0.45')} = 45 מאיות. ל-${M('0.45')} יש את כל מה שיש ל-${M('0.4')}, ועוד 5 מאיות. <div class="g-t"></div></div>
            </div>
            <div class="aha"><b>השיטה של מאיה עובדת תמיד.</b> כשמשלימים אפסים, ${M('0.8')} הופך ל-${M('0.80')}, ועכשיו ברור: 80 מאיות מול 75 מאיות. בעצם, כשהמספרים באותו אורך, אפשר להשוות אותם כמו מספרים רגילים.</div>
          </div>`;
        const two = (host, a, b) => { const d = document.createElement('div'); d.style.display = 'flex'; d.style.gap = '10px'; d.style.justifyContent = 'center'; d.style.direction = 'ltr';
          for (const [n, s] of [[a, N.str(N.of(a, 2))], [b, N.str(N.of(b, 2))]]) { const w = document.createElement('div'); w.style.textAlign = 'center'; w.appendChild(gridSVG(n, 110)); w.insertAdjacentHTML('beforeend', `<b>${M(s)}</b>`); d.appendChild(w); }
          el.querySelector(host).appendChild(d); };
        two('.g-n', 80, 75); two('.g-t', 40, 45);
        el.querySelectorAll('.opt').forEach(b => b.onclick = () => {
          const ok = b.dataset.v === 'm';
          b.classList.add(ok ? 'right' : 'wrong');
          if (!ok) { el.querySelector('.fbq').innerHTML = `<p class="muted">נסה שיטה של מישהו אחר על הדוגמה של החבר השני. היא עובדת?</p>`; return; }
          el.querySelectorAll('.opt').forEach(x => x.disabled = true);
          el.querySelector('.fbq').innerHTML = '<p><b>נכון!</b> והנה למה שני האחרים טועים:</p>';
          el.querySelector('.reveal').hidden = false; done();
        });
      },
    },
    {
      t: 'מכונת ההשוואה',
      r(el, done) {
        el.innerHTML = `
          <p class="do">כתוב שני מספרים, והמכונה תיישר אותם לפי הנקודה, תשלים אפסים (החיוורים) ותראה איפה ההבדל הראשון. נסה לפחות 3 זוגות.</p>
          <div class="controls" style="direction:ltr;justify-content:center">
            <input class="num-in a" value="0.8" dir="ltr" inputmode="decimal"><b style="font-size:1.4rem" class="rel">?</b><input class="num-in b" value="0.75" dir="ltr" inputmode="decimal">
          </div>
          <div class="controls chips" style="justify-content:center">${[['0.8', '0.75'], ['0.4', '0.45'], ['1.09', '1.1'], ['0.5', '0.50'], ['0.07', '0.1'], ['2.3', '2.035']].map(([a, b]) => `<button class="btn sm ex" data-a="${a}" data-b="${b}">${M(a + ' , ' + b)}</button>`).join('')}</div>
          <div class="cmp-out"></div>
          <div class="nl-host"></div>`;
        const ia = el.querySelector('.a'), ib = el.querySelector('.b');
        const tried = new Set();
        const svg = S.svg(LW(), 150, 'board framed'); el.querySelector('.nl-host').appendChild(svg);
        let nl = null;
        function run() {
          const a = ia.value.trim().replace(',', '.'), b = ib.value.trim().replace(',', '.');
          const va = N.parse(a), vb = N.parse(b), out = el.querySelector('.cmp-out');
          if (va === null || vb === null || va < 0 || vb < 0 || va >= 1000 * U || vb >= 1000 * U) { out.innerHTML = '<p class="muted">כתוב שני מספרים חיוביים (עד 999).</p>'; el.querySelector('.rel').textContent = '?'; return; }
          const norm = x => { const [i, f] = x.split('.'); return f ? String(Number(i || '0')) + '.' + f : String(Number(i || '0')); };
          const A = norm(a), B = norm(b);
          el.querySelector('.rel').textContent = va > vb ? '>' : va < vb ? '<' : '=';
          out.innerHTML = compareWhy(A, B);
          const hi = Math.max(N.num(va), N.num(vb)), top = hi <= 1 ? 1 : Math.ceil(hi);
          S.clear(svg); nl = new NumberLine(svg, { W: LW(), y: 80, a: 0, b: top, dense: top <= 3 });
          nl.setPins([{ v: N.num(va), cls: 'b', label: A }, { v: N.num(vb), cls: 'o', label: B, up: false }]);
          tried.add(A + '|' + B); if (tried.size >= 3) done();
        }
        ia.oninput = ib.oninput = run;
        el.querySelectorAll('.ex').forEach(b => b.onclick = () => { ia.value = b.dataset.a; ib.value = b.dataset.b; run(); });
        run();
      },
    },
    {
      t: 'דו-קרב: מי גדול?',
      r(el, done) {
        el.innerHTML = `<p>12 זוגות, וכל אחד מהם מלכודת. לחץ על המספר הגדול יותר, או על "שווים".</p><div class="qhost"></div>`;
        const kinds = ['long', 'short', 'zeroEnd', 'zeroMid', 'nines', 'whole', 'wzero', 'long', 'thou', 'short', 'nines9', 'long'];
        const order = shuffle(kinds);
        Quiz(el.querySelector('.qhost'), {
          rounds: 12, onDone: done,
          gen(i) {
            let [a, b] = trapPair(order[i]); if (Math.random() < .5) [a, b] = [b, a];
            const va = N.parse(a), vb = N.parse(b), right = va > vb ? 'a' : va < vb ? 'b' : '=';
            return {
              q: 'מי גדול יותר?', type: 'custom', maxTries: 2,
              mount(ans, submit) {
                ans.innerHTML = `<div class="duel" dir="ltr"><button class="dcard" data-v="a">${digitsHTML(a)}</button><button class="dcard" data-v="b">${digitsHTML(b)}</button></div>
                  <div style="text-align:center"><button class="eqbtn" data-v="=">הם שווים</button></div>`;
                const all = ans.querySelectorAll('[data-v]');
                all.forEach(x => x.onclick = () => {
                  const ok = x.dataset.v === right; x.classList.add(ok ? 'right' : 'wrong');
                  if (ok) all.forEach(y => y.disabled = true); else x.disabled = true;
                  submit(ok);
                });
                this.reveal = () => all.forEach(y => { y.disabled = true; if (y.dataset.v === right) y.classList.add('right'); });
              },
              hint: 'השלם אפסים בראש עד ששני המספרים באותו אורך אחרי הנקודה, ואז השווה.',
              explain: compareWhy(a, b),
            };
          },
        });
      },
    },
    {
      t: 'מיון: מהקטן לגדול',
      r(el, done) {
        el.innerHTML = `
          <p class="do">לחץ על המספרים לפי הסדר, <b>מהקטן ביותר לגדול ביותר</b>. שלושה סיבובים.</p>
          <div class="muted rnd"></div>
          <div class="slot-row"></div>
          <div class="sorter"></div>
          <div class="sfb"></div>
          <div class="nl-host"></div>`;
        const sets = [
          () => [`0.${ri(5, 9)}`, `0.${ri(1, 4)}${ri(1, 9)}`, `0.0${ri(1, 9)}`, `0.${ri(1, 4)}`, `1.${ri(1, 9)}`],
          () => { const d = ri(2, 7); return [`0.${d}`, `0.${d}${ri(1, 9)}`, `0.${d - 1}9`, `0.0${d}`, `0.${d}0${ri(1, 9)}`]; },
          () => { const w = ri(1, 3); return [`${w}.${ri(1, 9)}`, `${w}.0${ri(1, 9)}`, `${w - 1}.9${ri(1, 9)}`, `${w}.${ri(1, 9)}${ri(1, 9)}`, `${w}.00${ri(1, 9)}`]; },
        ];
        let r = 0, cards = [], placed = [];
        function round() {
          let s; do { s = sets[r](); } while (new Set(s.map(x => N.parse(x))).size < s.length);
          cards = shuffle(s); placed = [];
          el.querySelector('.rnd').textContent = `סיבוב ${r + 1} מתוך 3`;
          el.querySelector('.sfb').innerHTML = ''; el.querySelector('.nl-host').innerHTML = '';
          draw();
        }
        function draw() {
          el.querySelector('.slot-row').innerHTML = placed.map((c, k) => `${k ? '<span class="lt">&lt;</span>' : ''}<span class="scard placed">${digitsHTML(c)}</span>`).join('') || '<span class="muted">הקטן ביותר נכנס לכאן ראשון ←</span>';
          const left = cards.filter(c => !placed.includes(c));
          el.querySelector('.sorter').innerHTML = left.map(c => `<button class="scard" data-c="${c}">${digitsHTML(c)}</button>`).join('');
          el.querySelectorAll('.sorter .scard').forEach(b => b.onclick = () => choose(b.dataset.c, b));
        }
        function choose(c, btn) {
          const left = cards.filter(x => !placed.includes(x));
          const min = left.reduce((m, x) => N.parse(x) < N.parse(m) ? x : m);
          if (c !== min) {
            el.querySelector('.sfb').innerHTML = `<div class="qz-fb bad">${M(c)} עוד לא הקטן ביותר. תשווה אותו ל-${M(min)}:${compareWhy(c, min)}</div>`;
            btn.classList.add('wrong'); return;
          }
          placed.push(c); el.querySelector('.sfb').innerHTML = ''; draw();
          if (placed.length === cards.length) finishRound();
        }
        function finishRound() {
          const top = Math.ceil(Math.max(...cards.map(c => N.num(N.parse(c)))) + 1e-9) || 1;
          const svg = S.svg(LW(), 190, 'board framed'); el.querySelector('.nl-host').appendChild(svg);
          const nl = new NumberLine(svg, { W: LW(), y: 95, a: 0, b: top, dense: true });
          nl.setPins(placed.map((c, k) => ({ v: N.num(N.parse(c)), label: c, up: k % 2 === 0, cls: k % 2 ? 'o' : 'b' })));
          const last = r === 2;
          el.querySelector('.sfb').innerHTML = `<div class="qz-fb ok"><b>מסודר! ✓</b> כך הם נראים על הציר. ${last ? '' : ''}</div><div class="controls"><button class="btn primary nx">${last ? 'סיימתי ✓' : 'לסיבוב הבא ←'}</button></div>`;
          el.querySelector('.nx').onclick = () => { if (last) { confetti(); done(); el.querySelector('.nx').disabled = true; } else { r++; round(); } };
        }
        round();
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned">
          <h4>📏 השיטה שלא נכשלת</h4>
          <ol><li>משווים קודם את השלמים (מה שלפני הנקודה). מי שיש לו יותר שלמים, גדול יותר.</li>
          <li>אם השלמים שווים: <b>משלימים אפסים בסוף</b> עד שלשני המספרים יש אותו מספר ספרות אחרי הנקודה.</li>
          <li>משווים ספרה-ספרה, <b>משמאל לימין</b>. ההבדל הראשון מכריע.</li></ol>
          <ul>
            <li>אורך לא אומר כלום: ${M('0.8 > 0.75')} (הקצר גדול), אבל ${M('0.45 > 0.4')} (הארוך גדול).</li>
            <li>${M('0.5 = 0.50')}, אבל ${M('0.05 < 0.5')}.</li>
            <li>${M('0.09 < 0.1')}: תשע מאיות זה פחות מעשר מאיות.</li>
          </ul></div>`;
      },
    },
  ],
});
