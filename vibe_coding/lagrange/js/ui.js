export const $ = (s, root = document) => root.querySelector(s);

export function fmt(x, d = 2) {
  if (x === Infinity) return '∞';
  if (x === -Infinity) return '−∞';
  if (!Number.isFinite(x)) return '?';
  let s = x.toFixed(d);
  if (Number(s) === 0) s = (0).toFixed(d);
  return s.replace('-', '−');
}

// like fmt, without trailing zeros: 1.50 -> 1.5, 2.00 -> 2
export function fmtt(x, d = 2) {
  let s = fmt(x, d);
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s;
}

export function put(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function range(id, onInput, format = v => v) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + '-out');
  const sync = () => { if (out) out.textContent = format(+el.value); };
  el.addEventListener('input', () => { sync(); onInput(+el.value); });
  sync();
  return { el, set(v) { el.value = v; sync(); } };
}

export function segmented(id, onPick) {
  const btns = [...document.getElementById(id).querySelectorAll('button')];
  const pick = k => {
    btns.forEach(b => b.classList.toggle('on', b.dataset.k === k));
    onPick(k);
  };
  btns.forEach(b => b.addEventListener('click', () => pick(b.dataset.k)));
  return pick;
}

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
export const norm = a => Math.hypot(a[0], a[1]);
