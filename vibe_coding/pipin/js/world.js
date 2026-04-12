// world.js - מבנה נתונים מלא של ארץ התיכונה

const LOCATIONS = {
  // ═══════════════════════════════
  //        אזור השאייר והמערב
  // ═══════════════════════════════

  grey_havens: {
    id: "grey_havens",
    name: "נמלי האפור",
    region: "shire",
    description: "נמלי האלפים המפוארים על חוף הים הגדול. ספינות לבנות עם מפרשי כסף עוגנות לאורך רציפי אבן ארוכים. מגדלות דקיקים מתנשאים אל השמיים, ואור זהוב עוטף הכל באווירה של פרידה נצחית.",
    image: "images/locations/grey_havens.webp",
    exits: { north: null, south: null, east: "hobbiton", west: null },
    items: [],
    npcs: [],
    visited: false,
    ambient: "שאגת גלים, שחפים צורחים, שירת אלפים עמומה מרחוק"
  },

  hobbiton: {
    id: "hobbiton",
    name: "הוביטון",
    region: "shire",
    description: "כפר ציורי של חורי הוביטים בלב השאייר. גבעות ירוקות עם דלתות עגולות צבעוניות, גני ירק מטופחים, ושבילים מתפתלים בין הבתים. ריח של לחם אפוי ועשב מקטרת נישא באוויר.",
    image: "images/locations/hobbiton.webp",
    exits: { north: null, south: "old_forest", east: "bucklebury", west: "grey_havens" },
    items: ["pipe_weed", "walking_stick", "seed_cake"],
    npcs: ["gandalf", "gaffer_gamgee"],
    visited: false,
    ambient: "ציפורים מצייצות, רוח קלה מרשרשת בעלים"
  },

  bucklebury: {
    id: "bucklebury",
    name: "באקלברי",
    region: "shire",
    description: "מעבורת באקלברי חוצה את נהר ברנדיוויין הרחב. גבעות ירוקות משני צדי הנהר, ערבות על הגדות, ואווירה שלווה של בוקר ערפילי בשאייר.",
    image: "images/locations/bucklebury.webp",
    exits: { north: null, south: null, east: "bree", west: "hobbiton" },
    items: ["mushrooms"],
    npcs: [],
    visited: false,
    ambient: "מים זורמים, צפרדעים מקרקרות, רוח בערבות"
  },

  old_forest: {
    id: "old_forest",
    name: "היער הישן",
    region: "shire",
    description: "יער עתיק ואפל עם עצים עקומים ומפותלים. שורשים עבים חוצים את השביל, אור ירקרק מסתנן בקושי דרך החופה הצפופה. תחושה מעיקה שמישהו צופה בך.",
    image: "images/locations/old_forest.webp",
    exits: { north: "hobbiton", south: "tom_bombadil", east: null, west: null },
    items: [],
    npcs: [],
    visited: false,
    ambient: "ענפים חורקים, לחישות מוזרות, דממה מעיקה"
  },

  tom_bombadil: {
    id: "tom_bombadil",
    name: "בית טום בומבדיל",
    region: "shire",
    description: "בקתה עליזה בפינת יער ליד נחל. פרחים צהובים בכל מקום, דלת כחולה בהירה, ואור חם מבפנים. ניגוד מדהים בין חום הבית ליער האפל שמסביב.",
    image: "images/locations/tom_bombadil.webp",
    exits: { north: "old_forest", south: null, east: "bree", west: null },
    items: ["yellow_lily"],
    npcs: ["tom_bombadil_npc"],
    visited: false,
    ambient: "שירה עליזה, מים מפכים, ריח פרחים"
  },

  // ═══════════════════════════════
  //          הדרך מזרחה
  // ═══════════════════════════════

  bree: {
    id: "bree",
    name: "ברי",
    region: "eriador",
    description: "עיירה סואנת על פרשת דרכים, מוקפת חומה נמוכה. פונדק 'הסוס המשתולל' מאיר את הרחוב הראשי. כאן חיים בני-אדם והוביטים זה לצד זה.",
    image: "images/locations/bree.webp",
    exits: { north: null, south: null, east: "weathertop", west: "bucklebury" },
    items: ["ale_mug"],
    npcs: ["butterbur", "strider"],
    visited: false,
    ambient: "צחוק שיכורים מהפונדק, סוסים צוהלים"
  },

  weathertop: {
    id: "weathertop",
    name: "פסגת הרוח",
    region: "eriador",
    description: "מגדל שמירה הרוס על פסגת גבעה בודדה. חומות אבן שבורות, מדרגות עתיקות מובילות למעלה. ענני סערה כהים מתקבצים, ושטחי עשב רחבים נמתחים לכל עבר.",
    image: "images/locations/weathertop.webp",
    exits: { north: null, south: null, east: "trollshaws", west: "bree" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "רוח חזקה, עורבים קורקרים, דממה מאיימת"
  },

  trollshaws: {
    id: "trollshaws",
    name: "יערות הטרולים",
    region: "eriador",
    description: "יער סלעי עם אורנים ואבני טחב. גשר אבן מעל נחל גועש, ושלושה צורות אבן גדולות - טרולים מאובנים - מוסתרים בצמחייה.",
    image: "images/locations/trollshaws.webp",
    exits: { north: null, south: null, east: "rivendell", west: "weathertop" },
    items: ["troll_purse"],
    npcs: [],
    visited: false,
    ambient: "נחל גועש, רוח ביער, שקט של אורנים"
  },

  rivendell: {
    id: "rivendell",
    name: "ריבנדל",
    region: "eriador",
    description: "עמק האלפים המקודש. מבנים חינניים של אבן מגולפת בין מפלי מים, עצי סתיו בעלי זהב, גשרים קשתיים מעל נחלות קריסטליים. הרים מתנשאים מאחור.",
    image: "images/locations/rivendell.webp",
    exits: { north: null, south: "moria_west", east: null, west: "trollshaws" },
    items: ["lembas_bread"],
    npcs: ["elrond"],
    visited: false,
    ambient: "מפלי מים, שירת אלפים, ציפורים, שלווה עמוקה"
  },

  // ═══════════════════════════════
  //        הרי הערפל ומעבר
  // ═══════════════════════════════

  moria_west: {
    id: "moria_west",
    name: "שערי מוריה",
    region: "misty_mountains",
    description: "דלתות גמד עצומות חקוקות בצוק תלול. בריכה דוממת משתקפת לפניהן, הירח מאיר רונות כסופות חלשות על האבן. חשוך ודומם.",
    image: "images/locations/moria_west.webp",
    exits: { north: "rivendell", south: null, east: "moria", west: null },
    items: [],
    npcs: [],
    visited: false,
    ambient: "דממה מוחלטת, מים עומדים, רוח קרה מההרים"
  },

  moria: {
    id: "moria",
    name: "מכרות מוריה",
    region: "misty_mountains",
    description: "אולם תת-קרקעי עצום עם עמודי אבן מתנשאים בשורות שנעלמות לתוך החושך. אדריכלות גמדית חקוקה בדוגמאות גיאומטריות. זוהר כתום קלוש מעמקי האדמה.",
    image: "images/locations/moria.webp",
    exits: { north: null, south: null, east: "moria_east", west: "moria_west" },
    items: ["mithril_shirt"],
    npcs: [],
    visited: false,
    ambient: "הדים מרחוק, טפטוף מים, חושך מוחלט"
  },

  moria_east: {
    id: "moria_east",
    name: "גשר קאזאד-דום",
    region: "misty_mountains",
    description: "גשר אבן צר מעל תהום ללא תחתית. אור זורם מפתח לפניך, עמודים עתיקים חקוקים. תחושה של בריחה וסכנה.",
    image: "images/locations/moria_east.webp",
    exits: { north: null, south: "dimrill_dale", east: null, west: "moria" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "רוח עולה מהתהום, הדים של צעקות ישנות"
  },

  dimrill_dale: {
    id: "dimrill_dale",
    name: "עמק דימריל",
    region: "misty_mountains",
    description: "עמק הרים עם אגם דומם כמו מראה - מירורמיר. פסגות מושלגות משתקפות בצורה מושלמת במים. ערפל בוקר, עמודי גמד עתיקים לאורך הגדה.",
    image: "images/locations/dimrill_dale.webp",
    exits: { north: "moria_east", south: "lothlorien", east: null, west: null },
    items: [],
    npcs: [],
    visited: false,
    ambient: "שקט מוחלט, מים זכים, רוח הרים קרה"
  },

  lothlorien: {
    id: "lothlorien",
    name: "לות'לוריאן",
    region: "misty_mountains",
    description: "יער מוקסם של זהב. עצי מלורן ענקיים עם גזעי כסף ועלים זהובים. אור אתרי רך, במות עץ מורמות בצמרות העצים. איכות חלומית ונצחית.",
    image: "images/locations/lothlorien.webp",
    exits: { north: "dimrill_dale", south: "fangorn", east: null, west: null },
    items: ["elven_cloak", "phial_of_light"],
    npcs: ["galadriel"],
    visited: false,
    ambient: "שירה אלפית חלומית, אור זהוב, רוח עדינה בעלים"
  },

  // ═══════════════════════════════
  //            רוהאן
  // ═══════════════════════════════

  fangorn: {
    id: "fangorn",
    name: "יער פנגורן",
    region: "rohan",
    description: "יער קמאי עתיק. עצים ישנים להפליא מכוסים טחב וחזזית, סבך צפוף, אור ירוק מסונן. תחושה שהעצים צופים בך. עתיק ועמוק.",
    image: "images/locations/fangorn.webp",
    exits: { north: "lothlorien", south: "edoras", east: "dead_marshes", west: "isengard" },
    items: [],
    npcs: ["treebeard"],
    visited: false,
    ambient: "חריקת ענפים, לחישות עתיקות, צלילי טבע עמוקים"
  },

  isengard: {
    id: "isengard",
    name: "איזנגרד",
    region: "rohan",
    description: "מגדל שחור מאבן מבריקה - אורת'אנק - מתנשא ממתחם מוקף חומה עגולה. מכונות ברזל ובורות חורשים את הקרקע. עשן ואש תעשייתית, עננים כהים מאיימים.",
    image: "images/locations/isengard.webp",
    exits: { north: null, south: "helms_deep", east: "fangorn", west: null },
    items: ["palantir"],
    npcs: ["saruman"],
    visited: false,
    ambient: "פטישים על מתכת, אש בוערת, עשן שחור"
  },

  helms_deep: {
    id: "helms_deep",
    name: "עמק הלם",
    region: "rohan",
    description: "מבצר עצום הבנוי לתוך מצוק הר. חומת העומק נמתחת לרוחב עמק, הר בצורת קרן מאחור. לפידים לאורך המעקות, שמים סוערים ודרמטיים.",
    image: "images/locations/helms_deep.webp",
    exits: { north: "isengard", south: null, east: "edoras", west: null },
    items: ["rohirrim_horn"],
    npcs: [],
    visited: false,
    ambient: "רוח חזקה, להבות לפידים, הדי קרבות ישנים"
  },

  edoras: {
    id: "edoras",
    name: "אדוראס",
    region: "rohan",
    description: "אולם מדוסלד בגג הזהוב על ראש הגבעה. חומת עץ מסביב, דגלי רוכבי הסוסים מתנופפים ברוח חזקה. ערבות רוהאן הנרחבות נמתחות אל האופק.",
    image: "images/locations/edoras.webp",
    exits: { north: "fangorn", south: "minas_tirith", east: null, west: "helms_deep" },
    items: [],
    npcs: ["theoden", "eowyn"],
    visited: false,
    ambient: "רוח חזקה, סוסים צוהלים, דגלים מתנופפים"
  },

  // ═══════════════════════════════
  //            גונדור
  // ═══════════════════════════════

  ithilien: {
    id: "ithilien",
    name: "אית'יליאן",
    region: "gondor",
    description: "ארץ ירוקה ומיוערת עם בריכות נסתרות ומפלי מים. צמחייה שופעת, פרחי בר בין חורבות של גנים גונדוריאנים עתיקים. גן עדן סודי באור שמש חם ומנומר.",
    image: "images/locations/ithilien.webp",
    exits: { north: "dead_marshes", south: "minas_tirith", east: null, west: null },
    items: ["athelas"],
    npcs: ["faramir"],
    visited: false,
    ambient: "מפלי מים, ציפורים, ריח עשבי תיבול"
  },

  minas_tirith: {
    id: "minas_tirith",
    name: "מינאס טירית'",
    region: "gondor",
    description: "העיר הלבנה, בירת גונדור. שבע קומות של חומות לבנות נשגבות מתרוממות מול שדות הפלנור. מגדל אקת'ליון מנצנץ בשמש בפסגה.",
    image: "images/locations/minas_tirith.webp",
    exits: { north: "edoras", south: "pelargir", east: "osgiliath", west: null },
    items: ["gondor_banner"],
    npcs: ["denethor"],
    visited: false,
    ambient: "חצוצרות מרחוק, רוח שנושבת מהרי הצל"
  },

  osgiliath: {
    id: "osgiliath",
    name: "אוסגיליאת'",
    region: "gondor",
    description: "עיר חרבה משני צידי נהר רחב. גשרי אבן שבורים, מבנים מקומרים מתפוררים, עשבים שוטים. אווירה עצובה של מלחמה, שמיים אפורים ומעוננים.",
    image: "images/locations/osgiliath.webp",
    exits: { north: "ithilien", south: null, east: "minas_morgul", west: "minas_tirith" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "מים זורמים, רוח בחורבות, דממה מטרידה"
  },

  pelargir: {
    id: "pelargir",
    name: "פלארגיר",
    region: "gondor",
    description: "עיר נמל עם רציפי אבן, ספינות רבות-תרנים, מחסנים לאורך הגדה. נהר האנדואין זורם רחב ואפור. אדריכלות דרומית עם מבנים מקומרים.",
    image: "images/locations/pelargir.webp",
    exits: { north: "minas_tirith", south: null, east: null, west: "dol_amroth" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "גלים מטפחים על הרציף, עורבי ים, חבלים חורקים"
  },

  dol_amroth: {
    id: "dol_amroth",
    name: "דול אמרות'",
    region: "gondor",
    description: "טירה לבנה חופית על צוקים מעל הים. מגדלים גבוהים עם דגלי ברבור-ספינה בכחול וכסף. גלים מתנפצים למטה, ציפורי ים, אור ימי בהיר.",
    image: "images/locations/dol_amroth.webp",
    exits: { north: null, south: null, east: "pelargir", west: null },
    items: ["swan_knight_shield"],
    npcs: [],
    visited: false,
    ambient: "גלים מתנפצים, שחפים, רוח ים מלוחה"
  },

  // ═══════════════════════════════
  //            מורדור
  // ═══════════════════════════════

  dead_marshes: {
    id: "dead_marshes",
    name: "הביצות המתות",
    region: "mordor",
    description: "ביצה שוממת עצומה בדמדומים. שלוליות קופאות משקפות אור חולני חיוור, פנים רפאיים נראים בקושי מתחת לפני המים. להבות חיוורות, ערפל סמיך, מטריד לעומק.",
    image: "images/locations/dead_marshes.webp",
    exits: { north: null, south: "ithilien", east: "black_gate", west: "fangorn" },
    items: [],
    npcs: ["gollum"],
    visited: false,
    ambient: "בועות בביצה, רוח גונחת, דממת מוות"
  },

  black_gate: {
    id: "black_gate",
    name: "השער השחור",
    region: "mordor",
    description: "שערי ברזל ענקיים בין שני כתפי הר - המוראנון. שממה מושחרת לפניהם, ביצורי אורקים משני הצדדים. אפר ואבק באוויר, מאיים לחלוטין, שמים בצבע אדום כהה.",
    image: "images/locations/black_gate.webp",
    exits: { north: null, south: "mordor_plateau", east: null, west: "dead_marshes" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "תופי מלחמה מרחוק, שריקות אורקים, רוח שורפת"
  },

  minas_morgul: {
    id: "minas_morgul",
    name: "מינאס מורגול",
    region: "mordor",
    description: "מגדל עיר מושחת הזוהר באור זרחני חיוור וחולני. אדריכלות מעוותת, גשר מעל נחל מורעל, פרחים מתים. הדרך מתפתלת למעלה אל מעבר חשוך מאחור.",
    image: "images/locations/minas_morgul.webp",
    exits: { north: null, south: null, east: "cirith_ungol", west: "osgiliath" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "זמזום מטריד, אור חולני, ריח ריקבון"
  },

  cirith_ungol: {
    id: "cirith_ungol",
    name: "קירית' אונגול",
    region: "mordor",
    description: "מעבר הרים צר עם מגדל אבן מאיים. מדרגות תלולות חצובות בסלע, קורי עכביש בסדקים חשוכים. תחושה שצופים בך. אור ירח חיוור, צללים מאיימים.",
    image: "images/locations/cirith_ungol.webp",
    exits: { north: null, south: null, east: "mordor_plateau", west: "minas_morgul" },
    items: ["sting_dagger"],
    npcs: [],
    visited: false,
    ambient: "רחשושים בחושך, רוח יבבנית, שקט מאיים"
  },

  mordor_plateau: {
    id: "mordor_plateau",
    name: "רמת מורדור",
    region: "mordor",
    description: "מישור געשי שומם של סלע שחור סדוק. נהרות לבה מרחוק, אפר נופל כשלג. הר דום מעשן באופק, מגדל בראד-דור החשוך הרחק. נוף אפוקליפטי אדום.",
    image: "images/locations/mordor_plateau.webp",
    exits: { north: "black_gate", south: "mount_doom", east: "barad_dur", west: "cirith_ungol" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "רעמים, אדמה רועדת, חום חונק, אפר באוויר"
  },

  mount_doom: {
    id: "mount_doom",
    name: "הר דום",
    region: "mordor",
    description: "הר געש פעיל ענק המקיא אש ועשן. נהרות לבה זורמים במורדותיו. שביל צר מתפתל למעלה אל כניסה חשוכה. שמיים אדומים בוערים, אווירה אפוקליפטית.",
    image: "images/locations/mount_doom.webp",
    exits: { north: "mordor_plateau", south: null, east: null, west: null },
    items: [],
    npcs: [],
    visited: false,
    ambient: "רעם געשי, סלעים מתמוטטים, חום עצום"
  },

  barad_dur: {
    id: "barad_dur",
    name: "בראד-דור",
    region: "mordor",
    description: "מבצר חשוך בלתי-אפשרי בגובהו מברזל ואבן שחורה. עין בוערת גדולה בפסגתו. ברקים סביבו, מוקף מישורי אפר וצבאות אורקים. התגלמות הרוע המוחלט.",
    image: "images/locations/barad_dur.webp",
    exits: { north: null, south: null, east: null, west: "mordor_plateau" },
    items: [],
    npcs: [],
    visited: false,
    ambient: "רעמים מתמידים, שריקות נאזגול, אש ואפל"
  }
};

// ═══════════════════════════════
//            חפצים
// ═══════════════════════════════

const ITEMS = {
  pipe_weed: {
    id: "pipe_weed",
    name: "עשב מקטרת",
    description: "עשב מקטרת משובח מהשאייר הדרומי. הסוג האהוב על פיפין.",
    icon: "🌿",
    usable: true,
    useEffect: "calm"
  },
  walking_stick: {
    id: "walking_stick",
    name: "מקל הליכה",
    description: "מקל חזק מעץ אלון, מתאים למסע ארוך.",
    icon: "🪵",
    usable: true,
    useEffect: "utility"
  },
  seed_cake: {
    id: "seed_cake",
    name: "עוגת גרעינים",
    description: "עוגת גרעינים טרייה מהשאייר. טעימה ומשביעה.",
    icon: "🍰",
    usable: true,
    useEffect: "heal"
  },
  mushrooms: {
    id: "mushrooms",
    name: "פטריות",
    description: "פטריות טריות מהשאייר. האוכל האהוב על כל הוביט שמכבד את עצמו.",
    icon: "🍄",
    usable: true,
    useEffect: "heal"
  },
  yellow_lily: {
    id: "yellow_lily",
    name: "שושנה צהובה",
    description: "שושנת מים צהובה מגנו של טום בומבדיל. יש בה קסם עתיק.",
    icon: "🌼",
    usable: true,
    useEffect: "protection"
  },
  ale_mug: {
    id: "ale_mug",
    name: "כוס בירה",
    description: "כוס בירה מפונדק 'הסוס המשתולל'. חצי ליטר, כהובט אמיתי.",
    icon: "🍺",
    usable: true,
    useEffect: "calm"
  },
  troll_purse: {
    id: "troll_purse",
    name: "ארנק הטרול",
    description: "ארנק עור ישן שנמצא ליד הטרולים המאובנים. מכיל כמה מטבעות עתיקים.",
    icon: "👛",
    usable: false,
    useEffect: null
  },
  lembas_bread: {
    id: "lembas_bread",
    name: "לחם למבאס",
    description: "לחם דרכים אלפי. נגיסה אחת מספיקה ליום שלם של מסע.",
    icon: "🍞",
    usable: true,
    useEffect: "heal"
  },
  mithril_shirt: {
    id: "mithril_shirt",
    name: "שריון מית'ריל",
    description: "שריון גמדי מנצנץ, קל כנוצה וחזק כסלע. אין לו מחיר.",
    icon: "🛡️",
    usable: true,
    useEffect: "armor"
  },
  elven_cloak: {
    id: "elven_cloak",
    name: "גלימה אלפית",
    description: "גלימה מלות'לוריאן שמסתירה את הלובש ומשתלבת עם הסביבה.",
    icon: "🧥",
    usable: true,
    useEffect: "stealth"
  },
  phial_of_light: {
    id: "phial_of_light",
    name: "צלוחית האור",
    description: "צלוחית גלדריאל. מכילה את אור כוכב הערב. מאירה במקומות חשוכים.",
    icon: "✨",
    usable: true,
    useEffect: "light"
  },
  palantir: {
    id: "palantir",
    name: "פלנטיר",
    description: "אבן רואה. כדור כהה שמשקף דימויים מעולמות רחוקים... ומסוכנים.",
    icon: "🔮",
    usable: true,
    useEffect: "vision"
  },
  sting_dagger: {
    id: "sting_dagger",
    name: "פגיון ברוך",
    description: "פגיון אלפי מימי הממלכה הקדומה. זוהר בכחול כשאורקים קרובים.",
    icon: "🗡️",
    usable: true,
    useEffect: "weapon"
  },
  gondor_banner: {
    id: "gondor_banner",
    name: "דגל גונדור",
    description: "דגל העץ הלבן של גונדור. סמל התקווה של הממלכה.",
    icon: "🏳️",
    usable: false,
    useEffect: null
  },
  rohirrim_horn: {
    id: "rohirrim_horn",
    name: "קרן רוהירים",
    description: "קרן מלחמה של רוכבי רוהאן. קולה מהדהד בעמקים.",
    icon: "📯",
    usable: true,
    useEffect: "summon"
  },
  swan_knight_shield: {
    id: "swan_knight_shield",
    name: "מגן אביר הברבור",
    description: "מגן מפואר של אבירי דול אמרות', עם סמל ספינת הברבור.",
    icon: "🦢",
    usable: true,
    useEffect: "armor"
  },
  athelas: {
    id: "athelas",
    name: "אתלאס",
    description: "עשב מרפא המכונה 'עלה המלך'. ריחו מרפא ומחזק.",
    icon: "🌱",
    usable: true,
    useEffect: "heal"
  }
};

// ═══════════════════════════════
//            דמויות
// ═══════════════════════════════

const NPCS = {
  gandalf: {
    id: "gandalf",
    name: "גנדלף",
    race: "איסטאר (קוסם)",
    title: "גנדלף האפור",
    disposition: "friendly",
    personality: "חכם, מסתורי, סבלני אך נחרץ. יש לו חוש הומור מפתיע.",
    knowledge: "יודע על הטבעת, על סאורון, על תוכניות האויב. לא תמיד חולק הכל.",
    location: "hobbiton"
  },
  gaffer_gamgee: {
    id: "gaffer_gamgee",
    name: "האמון גאמג'י",
    race: "הוביט",
    title: "הגנן הזקן",
    disposition: "friendly",
    personality: "פשוט, ישיר, חובב גינון. לא מבין בהרפתקאות.",
    knowledge: "גינון, שמועות כפריות, מעט על בילבו.",
    location: "hobbiton"
  },
  tom_bombadil_npc: {
    id: "tom_bombadil_npc",
    name: "טום בומבדיל",
    race: "לא ידוע",
    title: "הקדמון",
    disposition: "friendly",
    personality: "עליז, שר תמיד, חידתי. בעל כוח עצום שאינו מנצל.",
    knowledge: "הכל על היער הישן, היסטוריה עתיקה, כוח על הטבעת.",
    location: "tom_bombadil"
  },
  butterbur: {
    id: "butterbur",
    name: "בארלימן בַּטֶרבּוּר",
    race: "אדם",
    title: "פונדקאי הסוס המשתולל",
    disposition: "friendly",
    personality: "שמנמן, עסוק, שכחן אך לבבי. תמיד רץ בין השולחנות.",
    knowledge: "שמועות מהדרך, אורחים מוזרים, מעט על סטריידר.",
    location: "bree"
  },
  strider: {
    id: "strider",
    name: "סטריידר",
    race: "אדם (דונדאין)",
    title: "אראגורן בן אראת'ורן",
    disposition: "guarded",
    personality: "שקט, עירני, מלכותי מתחת לחזות המחוספסת.",
    knowledge: "מכיר את השטחים הפראיים, מעקב, ריפוי בעשבי מרפא.",
    location: "bree"
  },
  elrond: {
    id: "elrond",
    name: "אלרונד",
    race: "חצי-אלף",
    title: "אדון ריבנדל",
    disposition: "friendly",
    personality: "חכם, מיושב, נושא את כובד אלפי שנים בחן. רופא ויועץ.",
    knowledge: "היסטוריה של ארץ התיכונה, הטבעות, אסטרטגיה, ריפוי.",
    location: "rivendell"
  },
  galadriel: {
    id: "galadriel",
    name: "גלדריאל",
    race: "אלפית",
    title: "גבירת לות'לוריאן",
    disposition: "friendly",
    personality: "נשגבה, רואה את הנסתר, דוברת בחידות. יפה ומאיימת כאחד.",
    knowledge: "עתיד, עבר, כוחות הרוע, תכונות הלב.",
    location: "lothlorien"
  },
  treebeard: {
    id: "treebeard",
    name: "זקן-עץ",
    race: "אנט",
    title: "פנגורן",
    disposition: "neutral",
    personality: "איטי, מתחשב, זוכר הכל. לא ממהר להחליט. עמוק כמו יער.",
    knowledge: "היסטוריה של ארץ התיכונה, צמחייה, סודות היער.",
    location: "fangorn"
  },
  saruman: {
    id: "saruman",
    name: "סרומאן",
    race: "איסטאר (קוסם)",
    title: "סרומאן הלבן",
    disposition: "hostile",
    personality: "ערמומי, יהיר, משכנע. קולו כמו דבש מורעל.",
    knowledge: "ידע עצום על טבעות כוח, מכונות מלחמה, שליטה.",
    location: "isengard"
  },
  theoden: {
    id: "theoden",
    name: "ת'אודן",
    race: "אדם",
    title: "מלך רוהאן",
    disposition: "friendly",
    personality: "מלך מזדקן אך נחוש. רוכב סוסים מיומן, אמיץ בקרב.",
    knowledge: "מלחמה, רוהאן, אבירות, היסטוריה של רוכבי הסוסים.",
    location: "edoras"
  },
  eowyn: {
    id: "eowyn",
    name: "אאוין",
    race: "אדם",
    title: "גבירת רוהאן",
    disposition: "friendly",
    personality: "אמיצה, נחושה, מסתירה עצב מאחורי חוזק. לוחמת נסתרת.",
    knowledge: "לחימה בחרב, רוהאן, ריפוי.",
    location: "edoras"
  },
  denethor: {
    id: "denethor",
    name: "דנת'ור",
    race: "אדם",
    title: "הסנשל של גונדור",
    disposition: "hostile",
    personality: "גאה, חשדן, שבור מאבדן בורומיר. הולך ומאבד שפיות.",
    knowledge: "הגנת גונדור, פוליטיקה, השתמש בפלנטיר.",
    location: "minas_tirith"
  },
  faramir: {
    id: "faramir",
    name: "פאראמיר",
    race: "אדם",
    title: "שר אית'יליאן",
    disposition: "friendly",
    personality: "עדין אך אמיץ, חובב ספרים ושירה. שונה מאביו.",
    knowledge: "אית'יליאן, הביצורים, טקטיקה, ידע אלפי.",
    location: "ithilien"
  },
  gollum: {
    id: "gollum",
    name: "גולום",
    race: "הוביט (מושחת)",
    title: "סמיאגול",
    disposition: "guarded",
    personality: "אומלל, פיצולי אישיות, אובססיבי. פעם היה הוביט.",
    knowledge: "מעברים נסתרים, מורדור, הטבעת.",
    location: "dead_marshes"
  }
};

export { LOCATIONS, ITEMS, NPCS };
