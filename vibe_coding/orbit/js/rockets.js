// rockets.js — נתוני הרקטות לסימולציית השיגור.
// מקורות: ויקיפדיה (טבלאות המפרט), מדריכי המשתמש של היצרנים ודו"חות הטיסה של נאס"א.
// ערכים שלא פורסמו רשמית מסומנים est (הערכה) ומוצגים כך גם בממשק.

const kN = 1000;

// מנוע מורכב מכמה מנועים שונים (למשל רפטור ים + רפטור ואקום): מחשבים Isp אפקטיבי
function combine(list) {
  let FV = 0, FS = 0, mdot = 0, mdotSL = 0;
  for (const e of list) {
    FV += e.n * e.vac; FS += e.n * e.sl;
    mdot += e.n * e.vac / (e.ispVac * 9.80665);
  }
  const ispVac = FV / (mdot * 9.80665);
  return { thrustVac: FV, thrustSL: FS, ispVac, ispSL: FS / (mdot * 9.80665) };
}

export const ROCKETS = {
  falcon9: {
    id: 'falcon9',
    name: 'Falcon 9 Block 5',
    nameHe: 'פלקון 9',
    country: 'ארה"ב · SpaceX',
    site: { name: 'קייפ קנוורל, פלורידה', lat: 28.56, lon: -80.58 },
    height: 69.8, diameter: 3.7,
    area: Math.PI * 1.85 ** 2,
    vertical: 7, kick: 3,
    maxPayloadGuess: 30000,
    stages: [
      { name: 'שלב ראשון', engines: '9 × Merlin 1D',
        core: { thrustSL: 7607 * kN, thrustVac: 8227 * kN, ispSL: 283, ispVac: 312, prop: 395700, dry: 25600 } },
      { name: 'שלב שני', engines: '1 × Merlin Vacuum',
        core: { thrustSL: 600 * kN, thrustVac: 981 * kN, ispSL: 200, ispVac: 348, prop: 92670, dry: 3900 } },
    ],
    fairing: { mass: 1900, sepAlt: 110000 },
    published: { leo: 22800, leoReuse: 17500, gto: 8300, gtoReuse: 5500 },
    defaultPayload: 15600, // אצוות סטארלינק טיפוסית
    defaultPayloadNote: 'כמו אצוות לווייני סטארלינק',
  },

  saturn5: {
    id: 'saturn5',
    name: 'Saturn V',
    nameHe: 'סטורן 5',
    country: 'ארה"ב · נאס"א',
    site: { name: 'מתחם 39A, מרכז החלל קנדי', lat: 28.61, lon: -80.60 },
    height: 110.6, diameter: 10.1,
    area: Math.PI * 5.05 ** 2,
    vertical: 12, kick: 2,
    maxPayloadGuess: 200000,
    stages: [
      { name: 'S-IC', engines: '5 × F-1',
        core: { thrustSL: 5 * 6770 * kN, thrustVac: 5 * 7770 * kN, ispSL: 263, ispVac: 304, prop: 2077000, dry: 137000 },
        cutAt: { t: 135.2, factor: 0.8 } }, // כיבוי המנוע המרכזי (CECO)
      { name: 'S-II', engines: '5 × J-2',
        core: { thrustSL: 3000 * kN, thrustVac: 4700 * kN, ispSL: 200, ispVac: 421, prop: 427000, dry: 43000 } },
      { name: 'S-IVB', engines: '1 × J-2',
        core: { thrustSL: 600 * kN, thrustVac: 1033 * kN, ispSL: 200, ispVac: 421, prop: 105300, dry: 15200 } },
    ],
    les: { mass: 4170, sepAfter: 197 }, // מגדל החילוץ נזרק כ־30 שניות אחרי הצתת S-II
    published: { leo: 140000, tli: 43500 },
    defaultPayload: 45700, // CSM + LM + SLA של אפולו 11
    defaultPayloadNote: 'חלליות אפולו 11 (CSM+LM)',
  },

  soyuz: {
    id: 'soyuz',
    name: 'Soyuz-2.1b',
    nameHe: 'סויוז־2.1b',
    country: 'רוסיה · רוסקוסמוס',
    site: { name: 'בייקונור, קזחסטן', lat: 45.92, lon: 63.34 },
    height: 46.3, diameter: 10.3,
    area: Math.PI * 1.48 ** 2 + 4 * Math.PI * 1.34 ** 2 * 0.6,
    vertical: 10, kick: 4,
    maxPayloadGuess: 12000,
    stages: [
      { name: 'ליבה + 4 מאיצים', engines: 'RD-108A + 4 × RD-107A',
        core: { thrustSL: 792.5 * kN, thrustVac: 990.2 * kN, ispSL: 255, ispVac: 319, prop: 90100, dry: 6545 },
        boosters: { count: 4, thrustSL: 838.5 * kN, thrustVac: 1021.3 * kN, ispSL: 262, ispVac: 319, prop: 39160, dry: 3784 } },
      { name: 'בלוק I', engines: '1 × RD-0124',
        core: { thrustSL: 200 * kN, thrustVac: 294.3 * kN, ispSL: 200, ispVac: 359, prop: 25400, dry: 2355 } },
    ],
    fairing: { mass: 1000, sepAlt: 100000, est: true },
    published: { leo: 8670 },
    defaultPayload: 7080, // חללית סויוז MS
    defaultPayloadNote: 'חללית סויוז MS עם צוות',
    defaultInclination: 51.6,
  },

  cz5: {
    id: 'cz5',
    name: 'Long March 5',
    nameHe: 'צעדה ארוכה 5',
    country: 'סין · CASC',
    site: { name: 'ון־צ\'אנג, האי היינאן', lat: 19.61, lon: 110.95 },
    height: 56.97, diameter: 5.0,
    area: Math.PI * 2.5 ** 2 + 4 * Math.PI * 1.675 ** 2 * 0.6,
    vertical: 10, kick: 3,
    maxPayloadGuess: 40000,
    stages: [
      { name: 'ליבה + 4 מאיצים', engines: '2 × YF-77 + 4 × (2 × YF-100)',
        core: { thrustSL: 1020 * kN, thrustVac: 1400 * kN, ispSL: 310, ispVac: 428, prop: 165300, dry: 21600, est: true },
        boosters: { count: 4, thrustSL: 2400 * kN, thrustVac: 2680 * kN, ispSL: 300, ispVac: 335, prop: 142800, dry: 13800, est: true } },
      { name: 'שלב שני', engines: '2 × YF-75D',
        core: { thrustSL: 100 * kN, thrustVac: 176.72 * kN, ispSL: 300, ispVac: 442.6, prop: 29600, dry: 6400, est: true } },
    ],
    fairing: { mass: 3000, sepAlt: 110000, est: true },
    published: { gto: 14000, tli: 8800 },
    defaultPayload: 8200, // צ'אנג'-אה 5
    defaultPayloadNote: 'כמו צ\'אנג\'-אה 5 בדרך לירח',
  },

  starship: {
    id: 'starship',
    name: 'Starship V3',
    nameHe: 'סטארשיפ V3',
    country: 'ארה"ב · SpaceX',
    site: { name: 'סטארבייס, טקסס', lat: 25.99, lon: -97.16 },
    height: 124.4, diameter: 9,
    area: Math.PI * 4.5 ** 2,
    vertical: 10, kick: 2,
    maxPayloadGuess: 250000,
    stages: [
      { name: 'Super Heavy', engines: '33 × Raptor 3',
        core: { thrustSL: 33 * 2746 * kN, thrustVac: 33 * 2912 * kN, ispSL: 330, ispVac: 350, prop: 3650000, dry: 275000, estDry: true } },
      { name: 'Starship', engines: '3 × Raptor 3 + 3 × Raptor Vacuum',
        core: { ...combine([{ n: 3, sl: 2746 * kN, vac: 2912 * kN, ispVac: 350 }, { n: 3, sl: 1500 * kN, vac: 3060 * kN, ispVac: 380 }]),
          prop: 1600000, dry: 100000, est: true },
        maxG: 3.5, minThrottle: 0.4 },
    ],
    published: { leo: 100000 },
    defaultPayload: 40000,
    defaultPayloadNote: 'כמו 26 לווייני סטארלינק V3 בטיסה 14 (משקל משוער)',
  },
};

export const ROCKET_ORDER = ['falcon9', 'starship', 'saturn5', 'soyuz', 'cz5'];
