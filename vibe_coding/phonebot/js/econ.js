// מחשבון כלכלת יחידה: כמה עולה, כמה נשאר, מתי מחזירים את ההשקעה.
const $ = id => document.getElementById(id);
const IDS = ['e-vol', 'e-bom', 'e-asm', 'e-ship', 'e-nre', 'e-cac', 'e-price'];
const money = v => (v < 0 ? '−' : '') + '$' + Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: v !== 0 && Math.abs(v) < 100 ? 2 : 0 });

function calc() {
  const v = Object.fromEntries(IDS.map(id => [id, +$(id).value]));
  for (const id of IDS) $(id + '-v').textContent = id === 'e-vol' ? (+v[id]).toLocaleString('en-US') : id === 'e-ship' ? v[id] + '%' : money(v[id]);
  const P = v['e-price'];
  const landed = (v['e-bom'] + v['e-asm']) * (1 + v['e-ship'] / 100);
  const nreUnit = v['e-nre'] / v['e-vol'];
  const channels = [
    { name: 'ישיר (אתר / קיקסטארטר)', fees: 0.05 * P + 6 + v['e-cac'], note: 'סליקה 5%, משלוח ואריזה $6, עלות רכישת לקוח' },
    { name: 'אמזון', fees: 0.15 * P + 5.5 + 0.5 * v['e-cac'], note: 'עמלה 15%, FBA כ-$5.5, פרסום' },
    { name: 'חנויות (קמעונאות)', fees: 0.5 * P + 0.08 * P, note: 'החנות לוקחת כ-50%, החזרות ושיווק בחנות 8%' },
  ];
  const rows = channels.map(c => {
    const contrib = P - landed - c.fees;
    const be = contrib > 0 ? Math.ceil(v['e-nre'] / contrib) : null;
    const net = contrib * v['e-vol'] - v['e-nre'];
    return `<tr><td>${c.name}<div class="hint">${c.note}</div></td>
      <td class="n">${money(contrib)}</td>
      <td class="n ${contrib > 0 ? 'good' : 'bad'}">${contrib > 0 ? Math.round(contrib / P * 100) + '%' : 'הפסד'}</td>
      <td class="n">${be ? be.toLocaleString('en-US') : '∞'}</td>
      <td class="n ${net > 0 ? 'good' : 'bad'}">${money(Math.round(net))}</td></tr>`;
  }).join('');
  $('e-out').innerHTML = rows;
  $('e-landed').textContent = money(landed);
  $('e-nreunit').textContent = money(nreUnit);
  $('e-mult').textContent = (P / landed).toFixed(1) + '×';
}
IDS.forEach(id => $(id).addEventListener('input', calc));
calc();
