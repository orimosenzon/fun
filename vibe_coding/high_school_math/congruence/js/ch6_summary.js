'use strict';
/* פרק 6: דף סיכום עם ציורים קטנים לכל משפט */

App.inits.ch6 = function () {
  const root = document.getElementById('summary');
  const T = { A: [22, 118], B: [178, 118], C: [118, 26] };
  const SIDE_V = { a: ['B', 'C'], b: ['A', 'C'], c: ['A', 'B'] };
  const others = v => ['A', 'B', 'C'].filter(x => x !== v);

  function mini(sel, bad) {
    const svg = S('svg', { viewBox: '0 0 200 140', class: 'mini' });
    const pts = [T.A, T.B, T.C], ctr = centroid(pts);
    D.poly(svg, pts, bad ? 'tri-gray' : 'tri-blue');
    sel.forEach(id => {
      if ('abc'.includes(id)) { const [p, q] = SIDE_V[id]; D.seg(svg, T[p], T[q], 'selside pink'); }
      else { const [p, q] = others(id); D.arc(svg, T[id], T[p], T[q], 1, 18, 'selarc pink', 'wedge-lock'); }
    });
    ['A', 'B', 'C'].forEach(n => D.label(svg, T[n], n, ctr, 13, 'lbl small'));
    return svg.outerHTML;
  }

  const CARDS = [
    { name: 'צ.צ.צ', sel: ['a', 'b', 'c'], txt: 'שלוש צלעות שוות בזוגות.', why: 'שני מעגלים נחתכים לכל היותר בשתי נקודות, והן תמונת מראה.' },
    { name: 'צ.ז.צ', sel: ['b', 'c', 'A'], txt: 'שתי צלעות והזווית <b>שביניהן</b>.', why: 'הזווית היא ציר: ברגע שהיא נעולה, הצלע השלישית נקבעת.' },
    { name: 'ז.צ.ז', sel: ['A', 'B', 'c'], txt: 'שתי זוויות והצלע <b>שביניהן</b>.', why: 'שתי קרניים נחתכות בנקודה אחת בלבד.' },
    { name: 'צ.צ.ז', sel: ['a', 'c', 'C'], txt: 'שתי צלעות והזווית <b>שמול הגדולה</b> מביניהן.', why: 'אם הזווית מול הצלע הגדולה, נקודת החיתוך השנייה בורחת אל מאחורי הקודקוד.' },
  ];
  const BAD = [
    { name: 'ז.ז.ז', sel: ['A', 'B', 'C'], txt: 'שלוש זוויות: קובעות צורה, לא גודל (משולשים דומים).' },
    { name: 'צ.צ.ז מול הקטנה', sel: ['a', 'c', 'A'], txt: 'יכולים לצאת שני משולשים שונים.' },
  ];

  root.innerHTML = `
    <div class="sum-grid">${CARDS.map(c => `<div class="sum-card">${mini(c.sel)}<h4>${c.name}</h4><p>${c.txt}</p><p class="why">למה? ${c.why}</p></div>`).join('')}</div>
    <h3 class="sub">ומה <b>לא</b> מספיק</h3>
    <div class="sum-grid two">${BAD.map(c => `<div class="sum-card bad">${mini(c.sel, true)}<h4>✗ ${c.name}</h4><p>${c.txt}</p></div>`).join('')}</div>
    <div class="card idea"><h4>💡 טיפ: ז.ז.צ</h4><p>שתי זוויות וצלע שלא ביניהן? מחשבים את הזווית השלישית (סכום הזוויות במשולש 180°), ואז זה ז.צ.ז.</p></div>
    <div class="card recipe"><h4>🧾 המתכון להוכחה</h4><ol>
      <li>מה צריך להוכיח? (שתי צלעות שוות? שתי זוויות שוות?)</li>
      <li>מצא שני משולשים, שכל אחד מכיל אחת מהן.</li>
      <li>מצא 3 זוגות שווים, עם נימוק לכל אחד.</li>
      <li>בחר משפט חפיפה, וכתוב את החפיפה בסדר הנכון של האותיות.</li>
      <li>מסקנה: צלעות (או זוויות) מתאימות במשולשים חופפים שוות.</li></ol></div>
    <div class="card"><h4>🧰 ארגז הכלים: מאיפה מגיעים זוגות שווים?</h4>
      <ul class="tools">
        <li><b>נתון</b>: כתוב בשאלה.</li>
        <li><b>צלע משותפת</b>: צלע ששייכת לשני המשולשים.</li>
        <li><b>זוויות קודקודיות</b>: שני ישרים נחתכים, והזוויות שמול זו את זו שוות.</li>
        <li><b>חוצה זווית</b>: מחלק זווית לשתי זוויות שוות.</li>
        <li><b>אמצע קטע</b> או <b>תיכון</b>: מחלק קטע לשני קטעים שווים.</li>
        <li><b>גובה</b> או <b>אנך</b>: זוויות של 90°.</li>
        <li><b>ישרים מקבילים</b>: זוויות מתחלפות ומתאימות שוות.</li>
        <li><b>חפיפה קודמת</b>: מה שקיבלת בחינם מחפיפה שכבר הוכחת.</li>
      </ul></div>
    <div class="controls noprint"><button class="btn" onclick="window.print()">🖨️ הדפס את הדף</button></div>`;
  App.markDone('ch6');
};
