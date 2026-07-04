// Bilingual UI strings (Hebrew default / English), persisted in localStorage.
// Note the <span dir="ltr"> wrappers in Hebrew help lines: arrow glyphs are
// bidi-neutral, so without them RTL layout renders "← / →" mirrored (arrows
// pointing inward) — the span pins the fragment to visual LTR order.
const STR = {
  he: {
    speed: 'מהירות', kmh: 'קמ"ש', agl: 'גובה מהקרקע', alt: 'רום', m: "מ'",
    assistOn: 'מייצב טיסה: פועל', assistOff: 'מייצב טיסה: כבוי',
    throttle: 'מצערת', lift: 'עילוי',
    rings: 'טבעות', time: 'זמן', best: 'שיא: ', secondsShort: "שנ'",
    wind: 'רוח',

    helpTitle: 'שליטה',
    helpLines: [
      '<span dir="ltr">← / →</span> — הטיה ופנייה',
      '<span dir="ltr">↓ / ↑</span> — הרמת / הורדת האף (עלרוד, כמו במטוס)',
      '<span dir="ltr">W / S</span> — מצערת הסילון האחורי',
      '<span dir="ltr">E / D</span> — עוצמת צינורות העילוי',
      'רווח — בוסט לסילון האחורי (מבער)',
      'Shift — פולס עילוי מרכזי (מנת דחף קצובה)',
      '<span dir="ltr">Z / X</span> — פולס גלגול שמאלה / ימינה',
      'C — מצלמה · R — איפוס · T — מייצב · M — שקט · L — English · H — עזרה',
    ],
    touchHelpLines: [
      "ג'ויסטיק — היגוי ועלרוד (משוך מטה כדי להרים את האף)",
      '▲/▼ מצערת ועילוי — מחזיקים',
      '🔥 — בוסט לסילון האחורי · ⬆ — פולס עילוי מרכזי',
      '↺ / ↻ — פולס גלגול (מנת דחף לכל הקשה)',
      '📷 מצלמה · T מייצב · ⟲ איפוס · 🔊 שקט · ? עזרה',
    ],

    subtitle: 'אופנוע סילון מעופף',
    intro: 'מנוע סילון אחורי דוחף קדימה, שלושה צינורות מופנים מטה מרחפים אותך באוויר.<br>' +
      'טוס דרך כל <span class="orange-t">הטבעות הכתומות</span> מהר ככל האפשר — ואל תתרסק.',
    startKeys1: '<span dir="ltr">←→</span> היגוי &nbsp;·&nbsp; <span dir="ltr">↓↑</span> עלרוד (אף) &nbsp;·&nbsp; WS מצערת &nbsp;·&nbsp; ED עילוי',
    startKeys2: 'רווח בוסט &nbsp;·&nbsp; Shift פולס עילוי &nbsp;·&nbsp; Z/X פולסי גלגול',
    touchKeys1: "ג'ויסטיק שמאלי — היגוי ועלרוד",
    touchKeys2: 'מימין — מצערת, עילוי, בוסטים',
    press: 'לחץ על מקש כלשהו כדי להמריא',
    touchPress: 'גע במסך כדי להמריא',
    portraitNote: 'מומלץ לשחק במצב אופקי (לרוחב) לחוויה הטובה ביותר',
    langBtn: 'English',

    reset: 'איפוס', ring: 'טבעת ',
    allRingsTime: 'כל הטבעות! זמן: ', seconds: 'שניות', newRecord: ' — שיא חדש!',
    muted: 'שקט', soundOn: 'סאונד פועל',
    assistMsgOn: 'מייצב טיסה: פועל', assistMsgOff: 'מייצב טיסה: כבוי — טיסה חופשית!',
    crash_water: 'שכשוך! נחתת במים', crash_ground: 'ריסוק! פגיעה חזקה מדי בקרקע',
    crash_flip: 'התהפכות!', backToPad: ' — חוזרים למשטח...',
  },
  en: {
    speed: 'Speed', kmh: 'km/h', agl: 'Height AGL', alt: 'Altitude', m: 'm',
    assistOn: 'Flight assist: ON', assistOff: 'Flight assist: OFF',
    throttle: 'Throttle', lift: 'Lift',
    rings: 'Rings', time: 'Time', best: 'Best: ', secondsShort: 's',
    wind: 'Wind',

    helpTitle: 'Controls',
    helpLines: [
      '← / → — bank & turn',
      '↓ / ↑ — nose up / down (pitch, airplane-style)',
      'W / S — rear jet throttle',
      'E / D — lift nozzle power',
      'Space — rear jet boost (afterburner)',
      'Shift — center lift pulse (metered impulse)',
      'Z / X — roll pulse left / right',
      'C — camera · R — reset · T — assist · M — mute · L — עברית · H — help',
    ],
    touchHelpLines: [
      'Joystick — steering & pitch (pull down to raise the nose)',
      '▲/▼ throttle & lift — press and hold',
      '🔥 — rear jet boost · ⬆ — center lift pulse',
      '↺ / ↻ — roll pulse (one impulse per tap)',
      '📷 camera · T assist · ⟲ reset · 🔊 mute · ? help',
    ],

    subtitle: 'Flying Jet Motorcycle',
    intro: 'A rear jet pushes you forward; three downward nozzles keep you hovering.<br>' +
      'Fly through all the <span class="orange-t">orange rings</span> as fast as you can — and don\'t crash.',
    startKeys1: '←→ steer &nbsp;·&nbsp; ↓↑ pitch (nose) &nbsp;·&nbsp; WS throttle &nbsp;·&nbsp; ED lift',
    startKeys2: 'Space boost &nbsp;·&nbsp; Shift lift pulse &nbsp;·&nbsp; Z/X roll pulses',
    touchKeys1: 'Left joystick — steering & pitch',
    touchKeys2: 'Right side — throttle, lift, boosts',
    press: 'Press any key to take off',
    touchPress: 'Tap the screen to take off',
    portraitNote: 'Landscape orientation is recommended',
    langBtn: 'עברית',

    reset: 'Reset', ring: 'Ring ',
    allRingsTime: 'All rings! Time: ', seconds: 's', newRecord: ' — new record!',
    muted: 'Muted', soundOn: 'Sound on',
    assistMsgOn: 'Flight assist: ON', assistMsgOff: 'Flight assist: OFF — free flight!',
    crash_water: 'Splash! You landed in the water', crash_ground: 'Crash! Hit the ground too hard',
    crash_flip: 'Flipped over!', backToPad: ' — returning to the pad...',
  },
};

let lang = 'he';
try { lang = localStorage.getItem('fable-lang') || 'he'; } catch { /* headless */ }
if (!STR[lang]) lang = 'he';

export function getLang() { return lang; }

export function setLang(l) {
  if (!STR[l]) return;
  lang = l;
  try { localStorage.setItem('fable-lang', l); } catch { /* headless */ }
  document.documentElement.lang = l;
  document.documentElement.dir = l === 'he' ? 'rtl' : 'ltr';
}

export function toggleLang() {
  setLang(lang === 'he' ? 'en' : 'he');
  return lang;
}

export const t = (k) => STR[lang][k] ?? STR.he[k] ?? k;

// apply persisted language to the document on load
setLang(lang);
