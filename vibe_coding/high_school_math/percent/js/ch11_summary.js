'use strict';
/* פרק 11: דף סיכום להדפסה, ובסוף פינה קטנה למורה עם ההתקדמות של עומר */

App.add({
  id: 'sum', short: 'דף סיכום', icon: '📋', desc: 'הכול בדף אחד, אפשר להדפיס',
  render(el) {
    const cards = [
      ['💯 מה זה אחוז', `<p>${M('p%')} = ${M(F('p', 100))}. "לכל מאה".</p><p>${M('25% = 0.25 = ' + '¼')} · ${M('12.5% = ⅛')} · ${M('150% = 1.5')}</p><p><b>תמיד לשאול: אחוז ממה?</b></p>`],
      ['🧮 אחוז מכמות', `<p>${M('p%')} מ-${M('x')} = ${M('p/100 × x')}.</p><p>בראש: אבני בניין ${M('10%, 5%, 1%')}.</p><p>החלפה: ${M('8%')} מ-50 = ${M('50%')} מ-8 = 4.</p>`],
      ['📊 כמה אחוז', `<p>${M('חלק ÷ שלם × 100')}.</p><p>${F(17, 20)} = ${F(85, 100)} = ${M('85%')}.</p><p>הסכום של כל החלקים: ${M('100%')}.</p>`],
      ['🔍 מוצאים את השלם', `<p>${M('שלם = חלק ÷ (p/100)')}.</p><p>${M('30%')} = 9 ← ${M('10%')} = 3 ← ${M('100%')} = 30.</p>`],
      ['📈 המכפיל', `<p>${M('+p%')} = ${M('× (1 + p/100)')} · ${M('−p%')} = ${M('× (1 − p/100)')}</p><p>שינויים ברצף: <b>כופלים מכפילים</b>. ${M('+50%, −50%')} = ${M('× 0.75')}.</p><p>חזרה אחורה: <b>מחלקים</b> במכפיל. אחרי ${M('−x')} צריך ${M('× 1/(1 − x)')} כדי לחזור.</p>`],
      ['⚖️ נקודות אחוז', `<p>מ-${M('4%')} ל-${M('5%')}: עלייה של נקודת אחוז אחת, שהיא ${M('25%')}.</p><p>סיכון יחסי מול מוחלט: "פי 2" מ-1 ל-2 מתוך 1000.</p>`],
      ['🌱 ריבית דריבית', `<p>${M('סכום × (1 + r)ⁿ')}. גדילה מעריכית.</p><p>חוק ה-72: מוכפל אחרי ${M('72 ÷ r')} שנים.</p><p>${M('1.01³⁶⁵ ≈ 37.8')}</p>`],
      ['🌀 פרדוקסים', `<p>אבטיח: מסתכלים על החלק שלא משתנה.</p><p>סימפסון: אחוז כולל הוא ממוצע משוקלל, התמהיל קובע.</p><p>ממוצע של אחוזים: מחברים כמויות, לא אחוזים.</p>`],
    ];
    const mine = Store.get('mine', {});
    el.innerHTML = `
      <h2><span class="num">11</span>דף סיכום</h2>
      <p>כל מה שלמדת, בדף אחד. <button class="btn sm noprint" onclick="window.print()">🖨️ להדפיס</button></p>
      <div class="cards">${cards.map(([h, b]) => `<div class="sum-card"><h4>${h}</h4>${b}</div>`).join('')}</div>
      <div class="card"><h4>🧭 שלוש שאלות לפני כל בעיה של אחוזים</h4>
        <ol><li><b>מה השלם?</b> כל אחוז הוא אחוז ממשהו. ואם יש כמה שלבים, השלם משתנה בכל שלב.</li>
        <li><b>מה המכפיל?</b> הפוך כל שינוי לכפל, ואז אפשר לשרשר ולחזור אחורה.</li>
        <li><b>מה לא משתנה?</b> החלק היבש, התרכיז, המרחק. לפעמים הוא המפתח.</li></ol></div>
      <details class="card noprint teacher"><summary><b>👨‍🏫 למורה: ההתקדמות של עומר</b></summary><div class="tbody"></div></details>`;
    const det = el.querySelector('.teacher');
    det.addEventListener('toggle', () => { if (det.open) this.fillTeacher(det.querySelector('.tbody')); });
  },
  fillTeacher(box) {
    const pre = Store.get('pre', {}) || {}, post = Store.get('post', null), sd = Store.get('sd', {}), mine = Store.get('mine', {});
    const rows = App.chapters.filter(c => c.steps).map(c => `<tr><td>${c.icon} ${c.short}</td><td>${(sd[c.id] || []).length} / ${c.steps.length}</td><td>${App.isDone(c.id) ? '✅' : ''}</td></tr>`).join('');
    const ans = PRETEST.map(t => { const a = (pre.ans || {})[t.id], b = post && post.ans[t.id]; const s = x => x === undefined ? '—' : `${esc(x)} ${pretestCorrect(t, x) ? '✅' : '❌'}`; return `<tr><td>${t.q}</td><td>${s(a)}</td><td>${post ? s(b) : '—'}</td></tr>`; }).join('');
    box.innerHTML = `<p class="muted small">הנתונים שמורים רק בדפדפן הזה (localStorage). אם עומר עובד ממכשיר אחר, ההתקדמות שלו נמצאת שם.</p>
      <table class="steps" style="width:100%;border-collapse:collapse"><tr><th style="text-align:right">פרק</th><th>תחנות</th><th>הושלם</th></tr>${rows}</table>
      <p><b>ההרפתקאות של עומר:</b> ${Store.get('omerScore', 0)} מתוך 10 בניסיון ראשון · <b>אתגר הגמר:</b> ${Store.get('best', 0)} מתוך 12</p>
      <h4>חידות הפתיחה</h4><table style="width:100%;border-collapse:collapse" class="steps"><tr><th style="text-align:right">שאלה</th><th>בהתחלה</th><th>בסוף</th></tr>${ans}</table>
      ${mine.q ? `<h4>הבעיה שעומר המציא</h4><div class="panel"><p>${esc(mine.q)}</p><p class="muted">${esc(mine.a || '')}</p></div>` : ''}
      <div class="controls"><button class="btn sm reset" style="color:var(--red)">איפוס כל ההתקדמות</button></div>`;
    box.querySelector('.reset').onclick = () => { if (confirm('למחוק את כל ההתקדמות והתשובות בדפדפן הזה?')) { try { localStorage.removeItem(Store.key); } catch (e) { /* */ } location.hash = ''; location.reload(); } };
  },
});
