'use strict';
/* פרק 12: דף סיכום להדפסה, ובסוף פינה קטנה למורה עם ההתקדמות של עידו */

App.add({
  id: 'ch12', short: 'דף סיכום', icon: '📋', desc: 'כל מה שלמדת בדף אחד, אפשר להדפיס',
  render(el) {
    const cards = [
      ['🍫 עשיריות', `<p>שלם שחתוך ל-10. ${M(`${M('0.3')} = ${F(3, 10)}`)} = שלוש עשיריות.</p><p>10 עשיריות = שלם: אחרי ${M('0.9')} בא ${M('1')}.</p>`],
      ['🟧 מאיות', `<p>שלם שחתוך ל-100. ${M('0.37')} = 37 מאיות = 3 עשיריות ו-7 מאיות.</p><p>${M('0.4 = 0.40')}, אבל ${M('0.05')} קטן פי 10 מ-${M('0.5')}.</p><p>שקל = 100 אגורות: ${M('1.5')} ₪ = 150 אגורות.</p>`],
      ['🏠 בית הספרות', `<div style="text-align:center">${digitsHTML('32.451')}</div><p>כל מקום שווה פי 10 מזה שמימינו. המרכז הוא השלמים: עשרות ↔ עשיריות, מאות ↔ מאיות.</p><p>קוראים לפי המקום האחרון: ${M('0.042')} = ארבעים ושתיים אלפיות.</p>`],
      ['🔍 ציר המספרים', `<p>בין כל שתי שנתות יש עוד 10. תמיד יש מספר באמצע: בין ${M('0.45')} ל-${M('0.46')} יש ${M('0.455')}.</p>`],
      ['⚖️ מי גדול?', `<p>1. משווים שלמים. 2. משלימים אפסים. 3. משווים משמאל לימין.</p><p>${M('0.8 > 0.75')} · ${M('0.45 > 0.4')} · ${M('0.09 < 0.1')}</p><p><b>האורך לא אומר כלום!</b></p>`],
      ['➕ חיבור וחיסור', `<p><b>מיישרים נקודות</b>, משלימים אפסים, ומחשבים מימין לשמאל.</p><p>10 בעמודה → מעבירים 1 שמאלה. חסר → פורטים מהעמודה משמאל.</p><p>${M('2.9 + 0.2 = 3.1')} · ${M('1 − 0.35 = 0.65')}</p>`],
      ['🚀 כפול 10, 100, 1000', `<p>הספרות זזות, הנקודה עומדת. ${M('× 10')}: מקום אחד שמאלה. ${M('÷ 10')}: מקום אחד ימינה.</p><p>${M('3.47 × 100 = 347')} · ${M('4 ÷ 100 = 0.04')}</p><p>ליחידה קטנה יותר כופלים: ${M('2.5')} מ׳ = 250 ס״מ.</p>`],
      ['✖️ כפל', `<p>כופלים בלי נקודות, ואז סופרים ספרות אחרי הנקודה בשני המספרים ביחד.</p><p>${M('0.3 × 0.4 = 0.12')} · ${M('1.2 × 0.03 = 0.036')}</p><p><b>כפל במספר קטן מ-1 מקטין.</b></p>`],
      ['➗ חילוק', `<p>במספר שלם: מחלקים כמו כסף, ופורטים את מה שנשאר. ${M('7.5 ÷ 3 = 2.5')}</p><p>במספר עשרוני: כופלים את שניהם ב-10 עד שהמחלק שלם. ${M('4.8 ÷ 0.6 = 48 ÷ 6 = 8')}</p><p><b>חילוק במספר קטן מ-1 מגדיל.</b></p>`],
      ['🔄 שברים ואחוזים', `<p>הופכים מכנה ל-10, 100 או 1000: ${M(`${F(3, 5)} = ${F(6, 10)} = ${M('0.6')}`)}.</p><p>${M(`${F(1, 2)} = ${M('0.5')}`)} · ${M(`${F(1, 4)} = ${M('0.25')}`)} · ${M(`${F(3, 4)} = ${M('0.75')}`)} · ${M(`${F(1, 5)} = ${M('0.2')}`)} · ${M(`${F(1, 8)} = ${M('0.125')}`)} · ${M(`${F(1, 3)} = ${M('0.333...')}`)}</p><p>אחוז = מאית: ${M('35% = 0.35')}.</p>`],
    ];
    el.innerHTML = `
      <h2><span class="num">12</span>דף סיכום</h2>
      <p>כל מה שלמדת, בדף אחד. <button class="btn sm noprint" onclick="window.print()">🖨️ להדפיס</button></p>
      <div class="cards">${cards.map(([h, b]) => `<div class="sum-card"><h4>${h}</h4>${b}</div>`).join('')}</div>
      <div class="card"><h4>🧭 שלוש שאלות ששואלים לפני כל תרגיל</h4>
        <ol><li><b>מה כל ספרה שווה?</b> תסתכל על המקום שלה ביחס לנקודה.</li>
        <li><b>בערך כמה זה צריך לצאת?</b> עגל ותעריך לפני שאתה מחשב.</li>
        <li><b>האם התשובה הגיונית?</b> כפל ב-${M('0.5')} צריך לתת חצי, חיסור צריך להקטין, וכן הלאה.</li></ol></div>
      <details class="card noprint teacher"><summary><b>👨‍🏫 למורה: ההתקדמות של עידו</b></summary><div class="tbody"></div></details>`;
    const det = el.querySelector('.teacher');
    det.addEventListener('toggle', () => { if (det.open) this.fillTeacher(det.querySelector('.tbody')); });
  },
  onShow() {},
  fillTeacher(box) {
    const pre = Store.get('pre', {}) || {}, post = Store.get('post', null), sd = Store.get('sd', {});
    const rows = App.chapters.slice(1, 12).map(c => `<tr><td>${c.icon} ${c.short}</td><td>${(sd[c.id] || []).length} / ${c.steps.length}</td><td>${App.isDone(c.id) ? '✅' : ''}</td></tr>`).join('');
    const ans = PRETEST.map(t => { const a = (pre.ans || {})[t.id], b = post && post.ans[t.id]; const s = (x) => x === undefined ? '—' : `${esc(x)} ${pretestCorrect(t, x) ? '✅' : '❌'}`; return `<tr><td>${t.q}</td><td>${s(a)}</td><td>${post ? s(b) : '—'}</td></tr>`; }).join('');
    box.innerHTML = `<p class="muted small">הנתונים שמורים רק בדפדפן הזה (localStorage). אם עידו עובד ממכשיר אחר, תראה שם את ההתקדמות שלו.</p>
      <table class="steps" style="width:100%;border-collapse:collapse"><tr><th style="text-align:right">פרק</th><th>תחנות</th><th>הושלם</th></tr>${rows}</table>
      <p><b>שיא באתגר הגמר:</b> ${Store.get('best', 0)} מתוך 15</p>
      <h4>שאלות הפתיחה</h4><table style="width:100%;border-collapse:collapse" class="steps"><tr><th style="text-align:right">שאלה</th><th>בהתחלה</th><th>בסוף</th></tr>${ans}</table>
      <div class="controls"><button class="btn sm reset" style="color:var(--red)">איפוס כל ההתקדמות</button></div>`;
    box.querySelector('.reset').onclick = () => { if (confirm('למחוק את כל ההתקדמות והתשובות בדפדפן הזה?')) { try { localStorage.removeItem(Store.key); } catch (e) { /* */ } location.hash = ''; location.reload(); } };
  },
});
