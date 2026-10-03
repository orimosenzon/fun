// settings.js — העדפות משתמש, נשמרות בדפדפן (localStorage) ומשודרות למי שמאזין
import { fmt } from './util.js';

const KEY = 'orbit-settings-v1';
export const DEFAULTS = {
  speedUnit: 'kmh',     // kmh | kms | ms
  timeZone: 'utc',      // utc | israel
  satSize: 1,           // מכפיל לגודל הלוויינים על המסך
  labels: true,
  clouds: true,
  quality: 'high',      // high | saver
};

export const settings = { ...DEFAULTS };
try { Object.assign(settings, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) { /* אין גישה לאחסון: נשארים עם ברירות המחדל */ }

const listeners = new Set();
export function onSettings(fn) { listeners.add(fn); return () => listeners.delete(fn); }
export function setSetting(k, v) {
  settings[k] = v;
  try { localStorage.setItem(KEY, JSON.stringify(settings)); } catch (e) { /* לא נשמר, אבל עדיין חל עכשיו */ }
  for (const fn of listeners) fn(k, v);
}

// ---------- מהירות ----------
export const SPEED_UNITS = {
  kmh: { label: 'קמ"ש', f: v => v * 3.6, digits: () => 0 },
  kms: { label: 'ק"מ/שנ׳', f: v => v / 1000, digits: x => x < 10 ? 2 : 1 },
  ms: { label: 'מ׳/שנ׳', f: v => v, digits: () => 0 },
};
export const speedUnit = () => SPEED_UNITS[settings.speedUnit] ?? SPEED_UNITS.kmh;
// מספר בלבד (בלי יחידה), מ-m/s
export function speedNum(mps) { const u = speedUnit(); const x = u.f(mps); return fmt(x, u.digits(x)); }
// מספר + יחידה
export function speedStr(mps) { return `${speedNum(mps)} ${speedUnit().label}`; }
// היחידה ה"אחרת" להצגה משנית (קמ"ש ↔ ק"מ/שנ׳)
export function altSpeedStr(mps) {
  return settings.speedUnit === 'kmh' ? `${fmt(mps / 1000, 2)} ק"מ/שנ׳` : `${fmt(mps * 3.6)} קמ"ש`;
}

// ---------- שעון ----------
export function clockStr(ms) {
  const d = new Date(ms);
  if (settings.timeZone === 'israel') {
    const s = d.toLocaleString('sv-SE', { timeZone: 'Asia/Jerusalem', hour12: false }).slice(0, 16);
    return s + ' שעון ישראל';
  }
  return d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
}
