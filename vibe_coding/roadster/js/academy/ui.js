// עזרים לבניית פקדים בווידג'טים
export function slider(parent, { label, min, max, step = 1, value, fmt = (v) => v, onInput }) {
  const d = document.createElement('div'); d.className = 'ctrl';
  const id = 'r' + Math.random().toString(36).slice(2);
  d.innerHTML = `<label for="${id}"><span>${label}</span><output></output></label><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
  const inp = d.querySelector('input'), out = d.querySelector('output');
  const upd = () => { out.textContent = fmt(+inp.value); onInput?.(+inp.value); };
  inp.addEventListener('input', upd);
  parent.appendChild(d);
  out.textContent = fmt(+inp.value);
  return { get value() { return +inp.value; }, set value(v) { inp.value = v; upd(); }, el: inp };
}
export function checkbox(parent, { label, value = false, onInput }) {
  const l = document.createElement('label'); l.className = 'check';
  l.innerHTML = `<input type="checkbox" ${value ? 'checked' : ''}> ${label}`;
  const inp = l.querySelector('input');
  inp.addEventListener('change', () => onInput?.(inp.checked));
  parent.appendChild(l);
  return { get value() { return inp.checked; } };
}
export function buttons(parent, items, { onPick, selected } = {}) {
  const d = document.createElement('div'); d.className = 'btns';
  const bs = items.map(([key, label]) => {
    const b = document.createElement('button'); b.textContent = label; b.dataset.k = key;
    if (key === selected) b.classList.add('on');
    b.onclick = () => { bs.forEach(x => x.classList.toggle('on', x === b)); onPick?.(key); };
    d.appendChild(b); return b;
  });
  parent.appendChild(d);
  return d;
}
export function readout(parent, rows) {
  const d = document.createElement('div'); d.className = 'readout';
  parent.appendChild(d);
  return {
    set(vals) { d.innerHTML = rows.map(([k, label]) => `<div class="kv"><span>${label}</span><b>${vals[k] ?? '—'}</b></div>`).join(''); },
  };
}
export const f = (x, d = 0) => (isFinite(x) ? x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—');

// לולאת אנימציה שרצה רק כשהרכיב גלוי
export function animate(el, fn) {
  let vis = false, last = 0, raf = 0;
  const loop = (t) => {
    raf = 0;
    if (!vis) return;
    const dt = Math.min(0.05, last ? (t - last) / 1000 : 0.016); last = t;
    fn(dt);
    raf = requestAnimationFrame(loop);
  };
  new IntersectionObserver((es) => {
    vis = es[0].isIntersecting;
    if (vis && !raf) { last = 0; raf = requestAnimationFrame(loop); }
  }, { rootMargin: '100px' }).observe(el);
}
