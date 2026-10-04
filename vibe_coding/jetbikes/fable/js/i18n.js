// Bilingual UI strings (Hebrew default / English), persisted in localStorage.
// Note the <span dir="ltr"> wrappers in Hebrew help lines: arrow glyphs are
// bidi-neutral, so without them RTL layout renders "← / →" mirrored (arrows
// pointing inward) — the span pins the fragment to visual LTR order.
const STR = {
  he: {
    speed: 'מהירות', kmh: 'קמ"ש', agl: 'גובה מהקרקע', alt: 'רום', m: "מ'",
    vspd: 'אנכית', ms: 'מ/ש',
    assistOn: 'מחשב טיסה: פועל', assistOff: 'מחשב טיסה: כבוי — ידני',
    throttle: 'מצערת', lift: 'עילוי',
    rings: 'טבעות', time: 'זמן', best: 'שיא: ', secondsShort: "שנ'",
    wind: 'רוח',

    helpTitle: 'שליטה — אתה מטה את הגוף, המחשב מטיס את הצינורות',
    helpLines: [
      '<span dir="ltr">← / →</span> — הטיית הגוף לפנייה',
      '<span dir="ltr">↑ / ↓</span> — רכינה קדימה = אף מטה וצלילה · נשענים אחורה = אף מעלה וטיפוס',
      '<span dir="ltr">W / S</span> — גריפ המצערת (הסילון האחורי)',
      '<span dir="ltr">E / D</span> — עלייה / ירידה — מחשב הטיסה מפעיל את צינורות העילוי לבד',
      'האופנוע טס לאן שהאף מצביע; אף ישר = הגובה נשמר אוטומטית',
      'רווח — מבער (בוסט) · Shift — פולס ניתור מרכזי',
      '<span dir="ltr">Z / X</span> — פולס גלגול שמאלה / ימינה',
      'T — מצב ידני · C — מצלמה · R — איפוס · M — שקט · L — English · H — עזרה',
    ],
    touchHelpLines: [
      "ג'ויסטיק — הטיית הגוף (דחוף קדימה = אף מטה וצלילה)",
      '▲/▼ מצערת · ▲/▼ גובה — מחזיקים; הגובה נשמר אוטומטית',
      '🔥 — מבער (בוסט) · ⬆ — פולס ניתור',
      '↺ / ↻ — פולס גלגול (מנת דחף לכל הקשה)',
      '📷 מצלמה · T ידני · ⟲ איפוס · 🔊 שקט · ? עזרה',
    ],

    subtitle: 'אופנוע סילון מעופף',
    intro: 'אתה מטה את הגוף — מחשב הטיסה מטיס את צינורות העילוי בשבילך.<br>' +
      'האופנוע טס לאן שהאף מצביע. טוס דרך כל <span class="orange-t">הטבעות הכתומות</span> מהר ככל האפשר — ואל תתרסק.',
    startKeys1: '<span dir="ltr">← → ↑ ↓</span> הטיית הגוף &nbsp;·&nbsp; WS מצערת &nbsp;·&nbsp; ED עלייה/ירידה',
    startKeys2: 'רווח בוסט &nbsp;·&nbsp; Shift ניתור &nbsp;·&nbsp; הגובה נשמר אוטומטית',
    touchKeys1: "ג'ויסטיק שמאלי — הטיית הגוף",
    touchKeys2: 'מימין — מצערת, עלייה/ירידה, בוסט',
    press: 'לחץ על מקש כלשהו כדי להמריא',
    touchPress: 'גע במסך כדי להמריא',
    portraitNote: 'מומלץ לשחק במצב אופקי (לרוחב) לחוויה הטובה ביותר',
    langBtn: 'English',

    reset: 'איפוס', ring: 'טבעת ',
    allRingsTime: 'כל הטבעות! זמן: ', seconds: 'שניות', newRecord: ' — שיא חדש!',
    muted: 'שקט', soundOn: 'סאונד פועל',
    assistMsgOn: 'מחשב טיסה: פועל — הגובה נשמר אוטומטית',
    assistMsgOff: 'מחשב טיסה: כבוי — שליטה ידנית מלאה (E/D = עוצמת עילוי)!',
    crash_water: 'שכשוך! נחתת במים', crash_ground: 'ריסוק! פגיעה חזקה מדי בקרקע',
    crash_flip: 'התהפכות!', backToPad: ' — חוזרים למשטח...',
  },
  en: {
    speed: 'Speed', kmh: 'km/h', agl: 'Height AGL', alt: 'Altitude', m: 'm',
    vspd: 'V-speed', ms: 'm/s',
    assistOn: 'Flight computer: ON', assistOff: 'Flight computer: OFF — manual',
    throttle: 'Throttle', lift: 'Lift',
    rings: 'Rings', time: 'Time', best: 'Best: ', secondsShort: 's',
    wind: 'Wind',

    helpTitle: 'Controls — you lean, the computer flies the nozzles',
    helpLines: [
      '← / → — lean into the turn',
      '↑ / ↓ — lean forward = nose down & dive · lean back = nose up & climb',
      'W / S — throttle grip (rear jet)',
      'E / D — climb / descend — the flight computer works the lift nozzles',
      'The bike goes where the nose points; nose level = altitude held',
      'Space — afterburner (boost) · Shift — center hop pulse',
      'Z / X — roll pulse left / right',
      'T — manual mode · C — camera · R — reset · M — mute · L — עברית · H — help',
    ],
    touchHelpLines: [
      'Joystick — body lean (push forward = nose down & dive)',
      '▲/▼ throttle · ▲/▼ climb/descend — hold; altitude is held for you',
      '🔥 — afterburner (boost) · ⬆ — hop pulse',
      '↺ / ↻ — roll pulse (one impulse per tap)',
      '📷 camera · T manual · ⟲ reset · 🔊 mute · ? help',
    ],

    subtitle: 'Flying Jet Motorcycle',
    intro: 'You lean — the flight computer flies the lift nozzles for you.<br>' +
      'The bike goes where the nose points. Fly through all the <span class="orange-t">orange rings</span> as fast as you can — and don\'t crash.',
    startKeys1: '← → ↑ ↓ body lean &nbsp;·&nbsp; WS throttle &nbsp;·&nbsp; ED climb/descend',
    startKeys2: 'Space boost &nbsp;·&nbsp; Shift hop &nbsp;·&nbsp; altitude is held automatically',
    touchKeys1: 'Left joystick — body lean',
    touchKeys2: 'Right side — throttle, climb/descend, boost',
    press: 'Press any key to take off',
    touchPress: 'Tap the screen to take off',
    portraitNote: 'Landscape orientation is recommended',
    langBtn: 'עברית',

    reset: 'Reset', ring: 'Ring ',
    allRingsTime: 'All rings! Time: ', seconds: 's', newRecord: ' — new record!',
    muted: 'Muted', soundOn: 'Sound on',
    assistMsgOn: 'Flight computer: ON — altitude held automatically',
    assistMsgOff: 'Flight computer: OFF — full manual (E/D = raw lift power)!',
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
