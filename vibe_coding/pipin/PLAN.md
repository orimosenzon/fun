# פיפין - משחק הרפתקאות בארץ התיכונה
## תוכנית מימוש מלאה

---

## 1. סקירה כללית

משחק הרפתקאות ויזואלי-טקסטואלי מודרני ואלגנטי, מבוסס דפדפן.
השחקן מגלם את **פיפין (פרגרין טוק)** בארץ התיכונה בתקופת שר הטבעות.

**עקרונות מנחים:**
- נאמנות לעולם של טולקין (גיאוגרפיה, דמויות, לור)
- חופש פעולה - העלילה לא עוקבת בדיוק אחרי הספר
- מנוע נרטיבי דינמי מבוסס Gemini API
- ממשק עברי (RTL) בעיצוב מודרני, בהיר ונקי - נוח לשימוש ומזמין
- תמונה ייחודית לכל מקום, מיוצרת מראש עם Nano Banana

---

## 2. מבנה הפרויקט

```
pipin/
├── index.html              # עמוד המשחק הראשי
├── style.css               # עיצוב מודרני בהיר RTL
├── js/
│   ├── game.js             # מנוע המשחק - ניהול מצב, ניווט, לוגיקה
│   ├── api.js              # תקשורת עם Gemini API
│   ├── ui.js               # עדכון ממשק, אפקטים, אירועים
│   ├── world.js            # מבנה נתונים: מקומות, קשרים, חפצים, דמויות
│   └── prompts.js          # System prompts ל-Gemini
├── images/
│   ├── locations/          # תמונת מקום לכל נקודה על המפה
│   │   ├── hobbiton.webp
│   │   ├── bucklebury.webp
│   │   ├── bree.webp
│   │   └── ...             # ~30 תמונות
│   └── npcs/               # תמונת פורטרט לכל דמות
│       ├── gandalf.webp
│       ├── strider.webp
│       ├── treebeard.webp
│       └── ...             # ~15 תמונות
├── tools/
│   └── generate_images.py  # סקריפט לייצור כל תמונות המקומות עם Nano Banana
├── PLAN.md                 # הקובץ הזה
└── README.md               # הוראות הפעלה
```

---

## 3. ממשק המשחק

### פילוסופיית העיצוב:
עיצוב מודרני, בהיר ומזמין - מרגיש כמו אפליקציית פרימיום. הדגש על קריאות, מרחב לבן, וחוויית משתמש חלקה.

- **צבעים**: רקע בהיר (off-white/cream), טקסט כהה, אקצנטים בגווני ירוק-יער וזהב חם שמזכירים ארץ התיכונה
- **טיפוגרפיה**: פונט עברי מודרני וקריא (כמו Heebo או Assistant), גדלים נוחים, line-height מרווח
- **פריסה**: CSS Grid/Flexbox מודרני, רספונסיבי, עם border-radius עדין ו-subtle shadows
- **אנימציות**: transitions חלקים בין מקומות, fade-in לטקסט חדש, hover effects עדינים
- **תמונות**: מוצגות ב-rounded corners עם צל עדין, תופסות מקום בולט

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   פיפין  ·  הוביטון                          🎒 תרמיל  │
│                                                         │
├───────────────────────────────┬─────────────────────────┤
│                               │                         │
│  ┌───────────────────────┐    │  אתה עומד בכיכר הכפר   │
│  │                       │    │  הוביטון. ריח של עשב    │
│  │                       │    │  מקטרת נישא באוויר.     │
│  │     תמונת המקום       │    │                         │
│  │     (rounded, shadow) │    │  גנדלף עומד ליד העגלה   │
│  │                       │    │  שלו ומחייך.            │
│  └───────────────────────┘    │                         │
│                               │  ╭─────────────────────╮│
│     ┌───┐                     │  │ גנדלף: "אה, פרגרין ││
│     │ צ │                     │  │ טוק! בדיוק בזמן."  ││
│  ┌──┼───┼──┐                  │  ╰─────────────────────╯│
│  │מ │ ● │ ד│                  │                         │
│  └──┼───┼──┘                  │  💡 דבר עם גנדלף       │
│     │ ר │                     │  💡 הסתכל מסביב        │
│     └───┘                     │  💡 לך דרומה           │
│                               ├─────────────────────────┤
│                               │  [מה תרצה לעשות?___] ▶ │
└───────────────────────────────┴─────────────────────────┘
```

### אזורי הממשק:
1. **סרגל עליון** - שם המשחק, שם המקום הנוכחי, וגישה לתרמיל (נפתח כ-drawer/modal)
2. **תמונת מקום** (שמאל-עליון) - מוצגת בפינות מעוגלות עם צל, מתחלפת באנימציית fade
3. **שושנת רוחות** (שמאל-תחתון) - כפתורים עגולים מינימליסטיים, כפתור מואפל = אין דרך
4. **חלון טקסט** (ימין) - תיאורים ודיאלוגים, עם typography נוחה ומרווחת. דיאלוגים מוצגים ב-bubble/card נפרד
5. **הצעות פעולה** (ימין-אמצע) - כפתורי chip לחיצים עם הצעות מה-AI
6. **שורת קלט** (ימין-תחתון) - input מודרני עם placeholder ברור וכפתור שליחה

### תצוגת דמויות (NPCs):
- מתחת לתמונת המקום, מוצגות תמונות פורטרט קטנות (אווטארים עגולים) של כל הדמויות שנמצאות כרגע באותו מקום
- לחיצה על פורטרט דמות פותחת אפשרות "דבר עם [שם]" בשורת הקלט
- כשדמות מדברת, הפורטרט שלה מופיע ליד ה-dialogue bubble בחלון הטקסט
- הפורטרטים מוצגים בגודל ~48px עם border עדין ושם הדמות מתחת

### עקרונות UX:
- **הכל לחיץ** - הצעות הפעולה מה-AI הן כפתורים, לא רק טקסט
- **היררכיה ברורה** - שם המקום בולט, תיאור בגודל נוח, דיאלוגים מובחנים ויזואלית
- **ריווח נדיב** - padding ו-margin נדיבים, אף אלמנט לא מרגיש צפוף
- **רספונסיבי** - עובד גם במסכים קטנים, layout מתאים את עצמו

---

## 4. מבנה נתונים - מפת העולם

### 4.1 מבנה מקום (Location)

```javascript
// world.js

const LOCATIONS = {
  hobbiton: {
    id: "hobbiton",
    name: "הוביטון",
    region: "shire",
    description: "כפר ציורי של חורי הוביטים בלב השאייר. גבעות ירוקות עם דלתות עגולות צבעוניות, גני ירק מטופחים, ושבילים מתפתלים בין הבתים. ריח של לחם אפוי ועשב מקטרת נישא באוויר.",
    image: "images/locations/hobbiton.webp",
    exits: {
      north: null,
      south: "old_forest",
      east: "bucklebury",
      west: "grey_havens"
    },
    items: ["pipe_weed", "walking_stick", "seed_cake"],
    npcs: ["gaffer_gamgee"],
    visited: false,
    ambient: "ציפורים מצייצות, רוח קלה מרשרשת בעלים"
  },

  bree: {
    id: "bree",
    name: "ברי",
    region: "eriador",
    description: "עיירה סואנת על פרשת דרכים, מוקפת חומה נמוכה. פונדק 'הסוס המשתולל' מאיר את הרחוב הראשי. כאן חיים בני-אדם והוביטים זה לצד זה.",
    image: "images/locations/bree.webp",
    exits: {
      north: null,
      south: null,
      east: "weathertop",
      west: "bucklebury"
    },
    items: ["ale_mug"],
    npcs: ["butterbur", "strider"],
    visited: false,
    ambient: "צחוק שיכורים מהפונדק, סוסים צוהלים"
  },

  minas_tirith: {
    id: "minas_tirith",
    name: "מינאס טירית׳",
    region: "gondor",
    description: "העיר הלבנה, בירת גונדור. שבע קומות של חומות לבנות נשגבות מתרוממות מול שדות הפלנור. מגדל אקת'ליון מנצנץ בשמש בפסגה.",
    image: "images/locations/minas_tirith.webp",
    exits: {
      north: "edoras",
      south: "pelargir",
      east: "osgiliath",
      west: null
    },
    items: ["gondor_banner"],
    npcs: ["denethor", "faramir"],
    visited: false,
    ambient: "חצוצרות מרחוק, רוח שנושבת מהרי הצל"
  }
};
```

### 4.2 מבנה חפץ (Item)

```javascript
const ITEMS = {
  pipe_weed: {
    id: "pipe_weed",
    name: "עשב מקטרת",
    description: "עשב מקטרת משובח מהשאייר הדרומי. הסוג האהוב על פיפין.",
    icon: "🌿",
    usable: true,
    useEffect: "calm"  // ה-AI יתאר את האפקט
  },
  walking_stick: {
    id: "walking_stick",
    name: "מקל הליכה",
    description: "מקל חזק מעץ אלון, מתאים למסע ארוך.",
    icon: "🪵",
    usable: true,
    useEffect: "utility"
  },
  sting_dagger: {
    id: "sting_dagger",
    name: "פגיון ברוך",
    description: "פגיון אלפי מימי הממלכה הקדומה. זוהר בכחול כשאורקים קרובים.",
    icon: "🗡️",
    usable: true,
    useEffect: "weapon"
  },
  lembas_bread: {
    id: "lembas_bread",
    name: "לחם למבאס",
    description: "לחם דרכים אלפי. נגיסה אחת מספיקה ליום שלם של מסע.",
    icon: "🍞",
    usable: true,
    useEffect: "heal"
  },
  palantir: {
    id: "palantir",
    name: "פלנטיר",
    description: "אבן רואה. כדור כהה שמשקף דימויים מעולמות רחוקים... ומסוכנים.",
    icon: "🔮",
    usable: true,
    useEffect: "vision"
  }
};
```

### 4.3 מבנה דמות (NPC)

```javascript
const NPCS = {
  gandalf: {
    id: "gandalf",
    name: "גנדלף",
    race: "איסטאר (קוסם)",
    title: "גנדלף האפור",
    portrait: "images/npcs/gandalf.webp",
    disposition: "friendly",
    personality: "חכם, מסתורי, סבלני אך נחרץ. יש לו חוש הומור מפתיע.",
    knowledge: "יודע על הטבעת, על סאורון, על תוכניות האויב. לא תמיד חולק הכל.",
    location: "hobbiton"  // מיקום התחלתי, יכול לנוע
  },
  strider: {
    id: "strider",
    name: "סטריידר",
    race: "אדם (דונדאין)",
    title: "אראגורן בן אראת'ורן",
    portrait: "images/npcs/strider.webp",
    disposition: "guarded",
    personality: "שקט, עירני, מלכותי מתחת לחזות המחוספסת.",
    knowledge: "מכיר את השטחים הפראיים, מעקב, ריפוי בעשבי מרפא.",
    location: "bree"
  },
  treebeard: {
    id: "treebeard",
    name: "זקן-עץ",
    race: "אנט",
    title: "פנגורן",
    portrait: "images/npcs/treebeard.webp",
    disposition: "neutral",
    personality: "איטי, מתחשב, זוכר הכל. לא ממהר להחליט. עמוק כמו יער.",
    knowledge: "היסטוריה של ארץ התיכונה, צמחייה, סודות היער.",
    location: "fangorn"
  },
  denethor: {
    id: "denethor",
    name: "דנת'ור",
    race: "אדם",
    title: "הסנשל של גונדור",
    portrait: "images/npcs/denethor.webp",
    disposition: "hostile",
    personality: "גאה, חשדן, שבור מאבדן בורומיר. הולך ומאבד שפיות.",
    knowledge: "הגנת גונדור, פוליטיקה, השתמש בפלנטיר.",
    location: "minas_tirith"
  }
};
```

### 4.4 מפת כל המקומות והקשרים

30 מקומות מחולקים ל-6 אזורים, על פי מפת טולקין:

```
                    ╔══════════════════════════════╗
                    ║      אזור השאייר והמערב      ║
                    ╚══════════════════════════════╝

grey_havens ──מזרח──> hobbiton ──מזרח──> bucklebury ──מזרח──> bree
                         │                                     │
                        דרום                                   │
                         ▼                                     │
                    old_forest                                 │
                         │                                     │
                        דרום                                   │
                         ▼                                     │
                    tom_bombadil ──────מזרח─────────────────────┘

                    ╔══════════════════════════════╗
                    ║       הדרך מזרחה             ║
                    ╚══════════════════════════════╝

bree ──מזרח──> weathertop ──מזרח──> trollshaws ──מזרח──> rivendell

                    ╔══════════════════════════════╗
                    ║      הרי הערפל ומעבר         ║
                    ╚══════════════════════════════╝

                    rivendell
                        │
                       דרום
                        ▼
                    moria_west ──מזרח──> moria ──מזרח──> moria_east
                                                             │
                                                            דרום
                                                             ▼
                                                        dimrill_dale
                                                             │
                                                            דרום
                                                             ▼
                                                        lothlorien

                    ╔══════════════════════════════╗
                    ║          רוהאן               ║
                    ╚══════════════════════════════╝

                    isengard ──מזרח──> fangorn
                        │                 │
                       דרום              דרום
                        ▼                 ▼
                    helms_deep ──מזרח──> edoras

                    ╔══════════════════════════════╗
                    ║          גונדור              ║
                    ╚══════════════════════════════╝

                                        ithilien
                                            │
                                           דרום
                                            ▼
    dol_amroth ──מזרח──> pelargir ──צפון──> minas_tirith ──מזרח──> osgiliath

                    ╔══════════════════════════════╗
                    ║          מורדור              ║
                    ╚══════════════════════════════╝

                    dead_marshes ──מזרח──> black_gate
                                              │
                                             דרום
                                              ▼
minas_morgul ──מזרח──> cirith_ungol ──מזרח──> mordor_plateau ──מזרח──> barad_dur
                                              │
                                             דרום
                                              ▼
                                          mount_doom
```

### 4.5 קשרים בין אזורים

```
lothlorien ──דרום──> fangorn        (מלות'לוריאן לרוהאן)
edoras ──דרום──> minas_tirith       (מרוהאן לגונדור)
fangorn ──מזרח──> dead_marshes      (דרך עוקפת מזרחית)
osgiliath ──מזרח──> minas_morgul    (מגונדור למורדור)
osgiliath ──צפון──> ithilien
ithilien ──צפון──> dead_marshes
dead_marshes ──דרום──> ithilien
```

---

## 5. ייצור תמונות עם Nano Banana

### מה זה Nano Banana?
[Nano Banana](https://ai.google.dev/gemini-api/docs/image-generation) הוא מודל ייצור תמונות של Google, מבוסס Gemini 2.5 Flash Image. הוא מאפשר ייצור תמונות מטקסט דרך ה-API של Gemini - אותו API שבו אנחנו כבר משתמשים למנוע המשחק.

### אסטרטגיית ייצור: סקריפט אוטומטי

נכתוב סקריפט Python שמייצר את כל 30 התמונות בלופ אחד, עם פרומפט עקבי שמבטיח סגנון אחיד:

```python
# tools/generate_images.py

from google import genai
import json
import time
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# סגנון בסיסי שחוזר בכל פרומפט - מבטיח עקביות ויזואלית
BASE_STYLE = """
Fantasy illustration in the style of Alan Lee and John Howe,
watercolor with ink details, muted earth tones,
Tolkien's Middle-earth aesthetic, atmospheric lighting,
wide landscape composition, no text, no characters in frame.
"""

# פרומפט ספציפי לכל מקום
LOCATION_PROMPTS = {
    "hobbiton": "A cozy hobbit village with round green doors set into grassy hillsides, flower gardens, a winding path, smoke rising from chimneys, warm afternoon light",

    "bucklebury": "Brandywine river crossing with a wooden ferry, rolling green hills of the Shire on both banks, willows along the riverbank, peaceful morning mist",

    "bree": "A medieval crossroads town at dusk, the Prancing Pony inn with warm light from windows, a low wooden palisade wall, muddy cobblestone streets, rain-slicked roofs",

    "grey_havens": "Elven harbor at sunset, tall white ships with silver sails, long stone quays reaching into a calm sea, towers of white stone, gulls flying overhead, melancholic golden light",

    "old_forest": "A dark ancient forest with gnarled twisted trees, thick roots crossing the path, dim green light filtering through dense canopy, oppressive and watchful atmosphere",

    "tom_bombadil": "A cheerful cottage in a forest clearing by a stream, yellow flowers everywhere, a bright blue door, warm firelight from within, contrast of cozy warmth against dark forest behind",

    "weathertop": "A ruined watchtower on a lone hilltop, broken stone walls, ancient steps leading up, dark storm clouds gathering, windswept grasslands stretching in all directions",

    "trollshaws": "A rocky pine forest with mossy boulders and a stone bridge over a rushing stream, three large stone shapes (petrified trolls) partially hidden by vegetation, dappled woodland light",

    "rivendell": "An elven valley sanctuary, graceful buildings of carved stone among waterfalls, autumn trees with golden leaves, arched bridges over crystal streams, mountains rising behind, ethereal morning light",

    "moria_west": "Massive ancient dwarf doors carved into a sheer cliff face, reflecting pool before them, moon shining overhead illuminating faint silvery runes on the stone, dark and still",

    "moria": "A vast underground hall with towering stone pillars in rows disappearing into darkness, dwarf architecture carved with geometric patterns, a faint orange glow from deep below, dust motes in shaft of light from above",

    "moria_east": "The eastern exit of a mountain, a narrow stone bridge over a bottomless chasm, light streaming in from an opening ahead, ancient carved pillars, sense of escape and danger",

    "dimrill_dale": "A mountain valley with a still mirror-like lake (Mirrormere), snow-capped peaks reflected perfectly in the water, morning mist, ancient dwarf pillars along the shore",

    "lothlorien": "An enchanted golden forest, massive mallorn trees with silver bark and golden leaves, soft ethereal light, elevated wooden platforms (flets) in the canopy, a dreamlike timeless quality",

    "fangorn": "An ancient primeval forest, incredibly old gnarled trees covered in moss and lichen, thick tangled undergrowth, green filtered light, a sense that the trees are watching, ancient and deep",

    "isengard": "A dark tower (Orthanc) of black glossy stone rising from a circular walled compound, iron machinery and pits scarring the ground, smoke and industrial fires, ominous dark clouds above",

    "edoras": "A golden-roofed mead hall (Meduseld) atop a hill, wooden palisade walls, horse-lord banners fluttering in strong wind, vast grasslands of Rohan stretching to the horizon, dramatic sky",

    "helms_deep": "A massive fortress built into a mountain cliff, the Deeping Wall stretching across a valley, a horn-shaped mountain behind, torches along the battlements, dramatic stormy sky",

    "minas_tirith": "A magnificent white stone city of seven tiers built into a mountainside, the Tower of Ecthelion gleaming at the summit, the great gates below, Pelennor Fields spreading before it, epic scale",

    "osgiliath": "A ruined city straddling a wide river, broken stone bridges, crumbling domed buildings, overgrown with weeds, a somber war-torn atmosphere, grey overcast sky",

    "pelargir": "A river port city with stone quays, many-masted ships, warehouses along the waterfront, the Anduin river flowing wide and grey, southern architecture with domed buildings",

    "dol_amroth": "A white coastal castle on cliffs above the sea, tall towers with blue and silver banners of a swan-ship, waves crashing below, seabirds, bright maritime light",

    "ithilien": "A green wooded land with hidden pools and waterfalls, lush vegetation, wildflowers among ruins of ancient Gondorian gardens, a secret paradise, warm dappled sunlight",

    "dead_marshes": "A vast desolate swamp at twilight, stagnant pools reflecting a pale sickly light, ghostly faces barely visible beneath the water surface, wisps of pale flame, thick fog, deeply unsettling",

    "black_gate": "Enormous iron gates set between two mountain spurs (Morannon), blackened wasteland before them, orc fortifications on either side, ash and dust in the air, utterly forbidding, dark red sky",

    "minas_morgul": "A corrupted tower city glowing with a sickly pale green phosphorescent light, twisted architecture, a bridge over a poisoned stream, dead flowers, the road winding up to a dark pass behind",

    "cirith_ungol": "A narrow mountain pass with a sinister stone tower, steep stairs carved into rock, cobwebs in dark crevices, a sense of being watched, pale moonlight, threatening shadows",

    "mordor_plateau": "A barren volcanic plateau of cracked black rock, rivers of lava in the distance, ash falling like snow, Mount Doom smoldering on the horizon, Barad-dur's dark tower far away, red-tinged hellscape",

    "mount_doom": "A massive active volcano belching fire and smoke, rivers of molten lava flowing down its slopes, a narrow path winding up to a dark entrance, red glowing sky, apocalyptic atmosphere",

    "barad_dur": "An impossibly tall dark fortress of iron and black stone, a great flaming eye at its summit, lightning striking around it, surrounded by ash plains and orc armies, ultimate evil incarnate"
}

OUTPUT_DIR = "../images/locations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for location_id, specific_prompt in LOCATION_PROMPTS.items():
    print(f"מייצר תמונה עבור: {location_id}...")

    full_prompt = f"{BASE_STYLE}\n\nScene: {specific_prompt}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=full_prompt
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(f"{OUTPUT_DIR}/{location_id}.webp")
                print(f"  ✓ נשמר: {location_id}.webp")
                break

        # המתנה קצרה בין בקשות למניעת rate limiting
        time.sleep(2)

    except Exception as e:
        print(f"  ✗ שגיאה ב-{location_id}: {e}")
        time.sleep(5)

print("\nסיום! כל התמונות נוצרו.")
```

### 5.2 פרומפטים לדמויות (NPCs)

```python
NPC_BASE_STYLE = """
Fantasy character portrait in the style of Alan Lee and John Howe,
watercolor with ink details, muted earth tones,
Tolkien's Middle-earth aesthetic, soft atmospheric lighting,
bust/shoulder portrait composition, painterly quality, no text.
"""

NPC_PROMPTS = {
    "gandalf": "An old wizard with a long grey beard and bushy eyebrows, wearing a pointed grey hat and grey robes, wise kind eyes with a hint of mischief, leaning on a wooden staff",

    "strider": "A rugged ranger with dark shoulder-length hair, weathered face with grey eyes, stubble beard, wearing a worn green-brown cloak with a silver star brooch, alert and noble bearing beneath rough exterior",

    "treebeard": "An ancient tree-creature (Ent), face formed from bark with deep eyes like dark pools, mossy beard of hanging lichen, gnarled branch-like features, impossibly old and wise",

    "denethor": "A proud aging lord in rich but dark robes, sharp intelligent eyes gone slightly mad with grief, grey hair, a silver circlet, gaunt aristocratic face, holding a seeing stone",

    "faramir": "A young nobleman with gentle but brave features, short brown hair, wearing leather and mail of Gondor, kind eyes, ranger's green cloak, resembles his father but softer",

    "butterbur": "A round jolly innkeeper with ruddy cheeks, balding head, wearing a stained apron, friendly but forgetful expression, wiping his hands on a cloth",

    "gaffer_gamgee": "An old hobbit gardener, short and stout, sun-weathered face, white wispy hair, wearing simple brown clothes and a worn hat, holding garden shears",

    "gollum": "A wretched emaciated creature with huge pale eyes, thin wispy hair, grey skin, crouching, long thin fingers, pitiful yet menacing, talking to himself",

    "saruman": "A tall imposing wizard with long white hair and beard, wearing white robes with subtle rainbow shimmer, piercing dark eyes, an air of corrupted grandeur, holding a black staff",

    "eowyn": "A young woman of Rohan with long golden hair, fair stern face, wearing a white dress with golden horse-lord embroidery, determined eyes hiding sadness, shield-maiden bearing",

    "theoden": "An aging king of Rohan with long grey-blond hair, golden crown, fur-trimmed royal robes, weary but dignified face, once-strong warrior now burdened by age and sorrow",

    "legolas": "An elf archer with long straight blond hair, ageless fair face, bright keen eyes, wearing woodland green and brown, elegant yet deadly, otherworldly grace",

    "gimli": "A stout dwarf warrior with a massive red-brown beard adorned with braids, iron helmet, chain mail, fierce loyal eyes, broad shoulders, holding a large double-headed axe",

    "boromir": "A proud warrior of Gondor with shoulder-length brown hair, strong square jaw, wearing armor with the white tree emblem, a large round shield, noble but troubled expression",

    "sam": "A young hobbit with curly brown hair, round honest face, sturdy build for a hobbit, wearing simple gardener's clothes, loyal determined eyes, carrying a cooking pot and pack"
}

NPC_OUTPUT_DIR = "../images/npcs"
os.makedirs(NPC_OUTPUT_DIR, exist_ok=True)

for npc_id, specific_prompt in NPC_PROMPTS.items():
    print(f"מייצר פורטרט עבור: {npc_id}...")

    full_prompt = f"{NPC_BASE_STYLE}\n\nCharacter: {specific_prompt}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=full_prompt
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(f"{NPC_OUTPUT_DIR}/{npc_id}.webp")
                print(f"  ✓ נשמר: {npc_id}.webp")
                break

        time.sleep(2)

    except Exception as e:
        print(f"  ✗ שגיאה ב-{npc_id}: {e}")
        time.sleep(5)
```

### עקרונות הפרומפטים:
1. **סגנון אחיד** - `BASE_STYLE` / `NPC_BASE_STYLE` חוזר בכל פרומפט: סגנון Alan Lee/John Howe, צבעי אדמה, אקוורל
2. **מקומות ללא דמויות** - "no characters in frame" - כדי שהתמונה תתאים לכל מצב
3. **דמויות כפורטרטים** - "bust/shoulder portrait" - קומפוזיציה אחידה שמתאימה לתצוגת אווטאר בממשק
4. **קומפוזיציה רחבה למקומות** - "wide landscape" - מתאים לתצוגה בממשק המשחק
5. **אטמוספרה** - כל מקום מקבל תאורה ואווירה ייחודית (הוביטון=חם, מורדור=אפוקליפטי)

### הרצה:
```bash
export GEMINI_API_KEY="your-key-here"
cd pipin/tools
python generate_images.py
```

עלות משוערת: ~30 תמונות × Gemini Flash = זניח (חלק מהמנוי הקיים)

---

## 6. מנוע המשחק (game.js)

### מצב המשחק:
```javascript
const gameState = {
  currentLocation: "hobbiton",
  inventory: ["pipe_weed", "walking_stick"],
  visitedLocations: ["hobbiton"],
  metNPCs: [],
  events: [],           // אירועים שקרו (לקונטקסט ל-AI)
  turnCount: 0,
  health: "טוב",       // מצב בריאות תיאורי
  conversationHistory: [] // היסטוריית שיחה ל-Gemini
};
```

### לולאת המשחק:
1. השחקן רואה תמונה + תיאור המקום
2. השחקן בוחר פעולה: ניווט (שושנת רוחות) או פקודת טקסט
3. הפעולה נשלחת ל-Gemini עם כל הקונטקסט
4. Gemini מחזיר JSON עם תיאור + שינויי מצב
5. הממשק מתעדכן (טקסט, תמונה, תרמיל, שושנת רוחות)
6. חוזר ל-1

### פקודות שהמנוע מזהה:
- **ניווט**: צפון/דרום/מזרח/מערב (גם דרך שושנת הרוחות)
- **בדיקה**: "הסתכל", "בדוק את...", "חפש"
- **אינטראקציה**: "דבר עם...", "קח...", "השתמש ב..."
- **מערכת**: "שמור", "טען", "עזרה", "מלאי"

---

## 7. חיבור Gemini (api.js)

### System Prompt:
```javascript
const SYSTEM_PROMPT = `
אתה מנוע משחק הרפתקאות המתרחש בארץ התיכונה של טולקין.
השחקן מגלם את פיפין (פרגרין טוק).

## על פיפין:
- הוביט צעיר מהשאייר, סקרן ופזיז
- אמיץ למרות גודלו הקטן
- אוהב אוכל, בירה ועשב מקטרת
- נאמן לחבריו עד מוות
- לפעמים נכנס לצרות בגלל סקרנותו

## כללי העולם:
- הקסם נדיר ושייך לקוסמים ולאלפים, לא להוביטים
- הטבעת היא כוח מושחת - פיפין לא נושא אותה אבל יודע עליה
- אורקים, טרולים ונאזגול הם סכנות אמיתיות
- הוביטים קטנים (~1.2 מטר) אבל זריזים ושקטים

## סגנון:
- כתוב בעברית, בסגנון סיפורי עשיר אך תמציתי
- תיאורים אטמוספריים ותמציתיים
- דיאלוגים נאמנים לאישיות הדמויות של טולקין
- אפשר הומור, במיוחד מפיפין

## פורמט תגובה - JSON בלבד:
{
  "description": "תיאור של מה קורה",
  "dialogue": "דיאלוג אם רלוונטי, או null",
  "options": ["הצעה 1", "הצעה 2", "הצעה 3"],
  "state_changes": {
    "location": "מיקום חדש או null",
    "inventory_add": [],
    "inventory_remove": [],
    "event": "תיאור קצר של אירוע שקרה, או null",
    "npc_met": "שם דמות שפגשנו, או null"
  }
}
`;
```

---

## 8. סדר מימוש

| שלב | משימה | תלויות |
|------|--------|---------|
| 1 | `world.js` - מבנה נתונים מלא (30 מקומות, חפצים, דמויות) | אין |
| 2 | `index.html` + `style.css` - ממשק מודרני בהיר עם כל האזורים | אין |
| 3 | `ui.js` - לוגיקת ממשק (שושנת רוחות, תרמיל, אפקט הקלדה) | שלב 2 |
| 4 | `game.js` - מנוע משחק (ניווט, מלאי, מצב) | שלבים 1+3 |
| 5 | `prompts.js` - הנחיות ל-AI | אין |
| 6 | `api.js` - חיבור Gemini | שלב 5 |
| 7 | שילוב הכל + בדיקה | שלבים 4+6 |
| 8 | `generate_images.py` - ייצור תמונות מקומות ודמויות | שלב 1 (רשימת מקומות ודמויות) |
| 9 | שילוב תמונות + ליטוש סופי | שלבים 7+8 |

---

## 9. בדיקות

- [ ] הממשק נטען ומוצג נכון ב-RTL
- [ ] שושנת רוחות מציגה רק כיוונים זמינים
- [ ] ניווט בין מקומות עובד ותמונה מתחלפת
- [ ] פקודות טקסט נשלחות ל-Gemini ותגובה מוצגת
- [ ] תרמיל מתעדכן בזמן אמת
- [ ] שמירה וטעינה מ-localStorage
- [ ] התגובות בעברית ונאמנות לטולקין
