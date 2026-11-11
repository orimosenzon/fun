// Language: Hebrew (default) or English. Chosen by ?lang= in the URL (so a shared link opens in
// that language), else the last choice saved on this device, else Hebrew.
// Static text in index.html carries data-i18n="key"; its Hebrew original stays in the HTML.
const KEY = 'foehn.lang';
const q = new URLSearchParams(location.search).get('lang');
let saved = null; try { saved = localStorage.getItem(KEY); } catch { /* private mode */ }
export let lang = q === 'en' || q === 'he' ? q : saved === 'en' ? 'en' : 'he';

const HE = {
  'ui.title': 'FÖHN · אופנוע סילון בעמק אלפיני',
  'reason.ground': 'פגעת בקרקע', 'reason.hard': 'נחיתה קשה מדי', 'reason.water': 'נפלת למים', 'reason.tree': 'עץ', 'reason.building': 'בניין',
  'reason.ring': 'פגעת בטבעת', 'reason.bounds': 'יצאת מגבולות העמק', 'reason.fuel': 'נגמר הדלק',
  'hud.best': 'השיא שלך: ', 'hud.noBest': 'עוד אין לך שיא כאן', 'hud.cp': 'נקודה', 'hud.penalty': 'עונשין', 'hud.pts': "נק'",
  'time.morning': 'בוקר', 'time.noon': 'צהריים', 'time.afternoon': 'אחר הצהריים', 'time.golden': 'שעת הזהב', 'time.dusk': 'דמדומים',
  'load.valley': 'טוען את העמק…', 'load.trees': 'שותל 150,000 עצים…', 'load.mountains': 'מגלף את ההרים…', 'load.shadows': 'מטיל צללים…',
  'load.forest': 'יער…', 'load.village': 'כפר ובקתות…', 'load.sun': 'מדליק את השמש…', 'load.engines': 'מחמם מנועים…',
  'hint.first': '<kbd>↓</kbd> להמריא · <kbd>W</kbd> להאיץ · <kbd>←</kbd> <kbd>→</kbd> לפנות · בסוף: <kbd>Shift</kbd> לבלום, <kbd>↑</kbd> לנחות על המעגל',
  'hint.free': 'טוס מהר ונמוך כדי לצבור נקודות. נחיתה על משטח ממלאת דלק',
  'card.gold': 'זהב: ', 'card.noClock': 'בלי שעון', 'card.best': 'שיא: ',
  'set.time': 'שעה: ', 'set.quality': 'איכות: ', 'q.low': 'נמוכה', 'q.medium': 'בינונית', 'q.high': 'גבוהה',
  'set.axis': 'ציר גובה: ', 'axis.inv': 'דחיפה = טיפוס', 'axis.norm': 'משיכה = טיפוס',
  'lang.other': 'English', 'share': 'שיתוף', 'share.copied': 'הקישור הועתק', 'share.text': 'FÖHN: אופנוע סילון בעמק אלפיני',
  'go': 'צא!', 'perfect': 'מרכז מושלם', 'landed': 'נחיתה!', 'landed.sub': 'להמריא ולהמשיך', 'missed': 'פספסת טבעת · +4 שניות',
  'warn.back': 'חזור לעמק!', 'warn.offpad': 'זה לא המעגל: להמריא ולנחות בתוכו', 'warn.heat': 'המבער התחמם', 'warn.stall': 'הזדקרות', 'warn.fuel': 'דלק אוזל',
  'medal.1': 'מדליית ארד', 'medal.2': 'מדליית כסף', 'medal.3': 'מדליית זהב', 'medal.none': 'בלי מדליה הפעם',
  'fin.newBest': 'שיא אישי חדש! קודם: ', 'fin.firstBest': 'שיא ראשון נקבע', 'fin.best': 'השיא שלך: ',
  'fin.crashes': 'התרסקויות: ', 'fin.misses': 'טבעות שפוספסו: ', 'fin.next': 'למדליה הבאה: ',
};

const EN = {
  'ui.title': 'FÖHN · A jet bike in an Alpine valley',
  'reason.ground': 'You hit the ground', 'reason.hard': 'Landing too hard', 'reason.water': 'Into the water', 'reason.tree': 'Tree', 'reason.building': 'Building',
  'reason.ring': 'You hit a ring', 'reason.bounds': 'You left the valley', 'reason.fuel': 'Out of fuel',
  'hud.best': 'Your best: ', 'hud.noBest': 'No record here yet', 'hud.cp': 'Gate', 'hud.penalty': 'penalty', 'hud.pts': 'pts',
  'time.morning': 'Morning', 'time.noon': 'Noon', 'time.afternoon': 'Afternoon', 'time.golden': 'Golden hour', 'time.dusk': 'Dusk',
  'load.valley': 'Loading the valley…', 'load.trees': 'Planting 150,000 trees…', 'load.mountains': 'Carving the mountains…', 'load.shadows': 'Casting shadows…',
  'load.forest': 'Forest…', 'load.village': 'Village and huts…', 'load.sun': 'Lighting the sun…', 'load.engines': 'Warming up the engines…',
  'hint.first': '<kbd>↓</kbd> take off · <kbd>W</kbd> speed up · <kbd>←</kbd> <kbd>→</kbd> turn · at the end: <kbd>Shift</kbd> to brake, <kbd>↑</kbd> to land on the circle',
  'hint.free': 'Fly fast and low to score points. Landing on a pad refuels',
  'card.gold': 'Gold: ', 'card.noClock': 'No clock', 'card.best': 'Best: ',
  'set.time': 'Time: ', 'set.quality': 'Quality: ', 'q.low': 'low', 'q.medium': 'medium', 'q.high': 'high',
  'set.axis': 'Pitch: ', 'axis.inv': 'push = climb', 'axis.norm': 'pull = climb',
  'lang.other': 'עברית', 'share': 'Share', 'share.copied': 'Link copied', 'share.text': 'FÖHN: a jet bike in an Alpine valley',
  'go': 'GO!', 'perfect': 'Dead centre', 'landed': 'Landed!', 'landed.sub': 'Take off and carry on', 'missed': 'Missed a ring · +4 s',
  'warn.back': 'Back to the valley!', 'warn.offpad': 'Not on the circle: take off and land inside it', 'warn.heat': 'Afterburner overheated', 'warn.stall': 'Stall', 'warn.fuel': 'Low fuel',
  'medal.1': 'Bronze medal', 'medal.2': 'Silver medal', 'medal.3': 'Gold medal', 'medal.none': 'No medal this time',
  'fin.newBest': 'New personal best! Previous: ', 'fin.firstBest': 'First record set', 'fin.best': 'Your best: ',
  'fin.crashes': 'Crashes: ', 'fin.misses': 'Rings missed: ', 'fin.next': 'Next medal: ',
  // events
  'ev.first.name': 'First Flight', 'ev.first.tag': 'Take off, six wide rings and a landing in the village square',
  'ev.lake.name': 'Lake Sprint', 'ev.lake.tag': 'Low over the water, around the island and back to the pier',
  'ev.gorge.name': 'The Gorge', 'ev.gorge.tag': 'A 900 m climb up a narrow gorge to the upper lake',
  'ev.pass.name': 'Mountain Pass', 'ev.pass.tag': 'Thin air: over the ridge at 2,300 m and a dive back into the valley',
  'ev.courier.name': 'Mountain Courier', 'ev.courier.tag': 'Three precision landings: the ledge hut, the alp hut and the village. Your route, your call',
  'ev.tour.name': 'The Grand Tour', 'ev.tour.tag': 'The whole valley: climb to the pass, a fast run down the gorge, the village and a lap of the island',
  'ev.free.name': 'Free Flight', 'ev.free.tag': 'No clock. Points for flying fast and close to the ground',
  // static page (data-i18n)
  'sub': 'A jet bike with tilting engines, one Alpine valley, six events',
  'eng.thrust': 'thrust', 'eng.noz': 'nozzles', 'eng.heat': 'burner heat', 'eng.fuel': 'fuel',
  'b.help': 'Keys & help', 'p.title': 'Paused', 'p.resume': 'Resume', 'p.restart': 'Restart', 'p.menu': 'Main menu', 'p.keys': 'Keys',
  'f.retry': 'Again <kbd>R</kbd>', 'f.next': 'Next event', 'f.menu': 'Menu',
  'h.title': 'How to fly',
  'h.intro': 'The flight computer does the hard part: you ask for a direction, height and speed, and it tilts the engines, moves the flaps and splits the thrust. At low speed the bike hovers on its jets; above about 220 km/h the wing carries it.',
  'h.turn': 'Turn. The bike banks by itself; a very hard turn costs speed',
  'h.pitch': 'Pull to climb, push to descend (can be flipped in the settings)',
  'h.speed': 'Target speed up and down. The computer holds it',
  'h.boost': 'Afterburner. About 7 seconds until it overheats; cools faster in fast flight',
  'h.brake': 'Brake: air brake and nozzles forward, down to a hover',
  'h.strafe': 'Slide sideways in a hover, for a precise landing',
  'h.cam': 'Camera: chase, close, rider’s eyes, cinematic',
  'h.misc': 'Restart · Pause · Mute',
  'h.land': 'Landing: brake to a hover above the circle, then push <kbd>↑</kbd> gently. The computer slows the descent near the ground and cuts the engines on touchdown; the event ends once you are resting inside the circle. Gamepads are supported.',
  'h.close': 'Got it',
  't.up': 'fast', 't.down': 'slow', 't.boost': 'burn', 't.brake': 'brake',
};

const DICT = { he: HE, en: EN };
export const t = (k) => DICT[lang][k] ?? HE[k] ?? k;
export const evName = (ev) => (lang === 'en' && EN[`ev.${ev.id}.name`]) || ev.name;
export const evTag = (ev) => (lang === 'en' && EN[`ev.${ev.id}.tag`]) || ev.tag;

// the link to this page in the current language
export function langUrl() { const u = new URL(location.href); u.searchParams.set('lang', lang); return u.href; }

const orig = new Map();
export function applyStatic() {
  const root = document.documentElement; root.lang = lang; root.dir = lang === 'en' ? 'ltr' : 'rtl';
  document.title = t('ui.title');
  for (const el of document.querySelectorAll('[data-i18n]')) {
    if (!orig.has(el)) orig.set(el, el.innerHTML);
    el.innerHTML = lang === 'he' ? orig.get(el) : (EN[el.dataset.i18n] ?? orig.get(el));
  }
}

export function setLang(l) {
  lang = l;
  try { localStorage.setItem(KEY, l); } catch { /* private mode */ }
  history.replaceState(null, '', langUrl());
  applyStatic();
}
applyStatic();
