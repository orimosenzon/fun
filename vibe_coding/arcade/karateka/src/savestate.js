// Save states: a snapshot of the whole machine (CPU, memory, language card, drive, video
// switches) and of our extensions, as plain JSON that can sit in localStorage or in a file.
// The disk itself is not in the snapshot; loading needs a machine booted from the same disk.
const VERSION = 1;
const CPU = ['a', 'x', 'y', 'sp', 'pc', 'n', 'v', 'd', 'i', 'z', 'c', 'cycles', 'illegal'];
const MACHINE = ['lcRead', 'lcWrite', 'lcBank2', 'lcPre', 'key', 'paddleStart', 'inGame', 'keyboardMode'];
const MEMORY = ['ram', 'lcRam', 'lcBank1'];
const EXTENSIONS = ['mawashi', 'flip'];

const b64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 8192) s += String.fromCharCode(...u8.subarray(i, i + 8192)); return btoa(s); };
const unb64 = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
const pick = (o, keys) => Object.fromEntries(keys.map((k) => [k, o[k]]));

export function saveState(m) {
  const s = { v: VERSION, cpu: pick(m.cpu, CPU), machine: pick(m, MACHINE), video: { ...m.video }, drive: { ...m.drive }, buttons: [...m.buttons], ext: {} };
  for (const k of MEMORY) s[k] = b64(m[k]);
  for (const k of EXTENSIONS) if (m[k]) s.ext[k] = { ...m[k] };
  return s;
}

export function loadState(m, s) {
  if (!s || s.v !== VERSION || !s.cpu || !s.ram) throw new Error('זה לא קובץ מצב שמור של המשחק');
  for (const k of MEMORY) { const d = unb64(s[k]); if (d.length !== m[k].length) throw new Error('קובץ המצב פגום'); m[k].set(d); }
  Object.assign(m.cpu, s.cpu); Object.assign(m, s.machine);
  // the extensions keep references to these objects, so they are filled in, not replaced
  Object.assign(m.video, s.video); Object.assign(m.drive, s.drive); m.buttons.splice(0, m.buttons.length, ...s.buttons);
  for (const k of EXTENSIONS) if (m[k] && s.ext[k]) Object.assign(m[k], s.ext[k]);
  m.keyQueue = []; m.speaker = [];
}
