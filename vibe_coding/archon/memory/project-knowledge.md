# Archon (ארקון) — ידע כללי על הפרויקט

## מטרה
שחזור המשחק **Archon: The Light and the Dark** (Free Fall Associates, 1983) — כמשחק דפדפן מודרני, נאמן למקור מהקומודור 64, עם ריבוי שחקנים אונליין.

## החלטות עקרוניות (סשן ראשון, 2026-05-20)
- **סטאק טכני**: HTML5 Canvas + Vanilla JS (frontend), Node.js + Express + Socket.IO (backend)
- **היקף**: ארקון מלא מהיום הראשון — לוח אסטרטגי + קרבות action + מחזור אור/חושך + 18 כלים לכל צד + לחשים
- **ריבוי שחקנים**: Online multiplayer (Socket.IO), עם lobby למאצ'מייקינג
- **גרפיקה**: Pixel-art מודרני (HD) — סגנון נאמן ל-1983 אבל ברזולוציה גבוהה ועם אנימציות חלקות

## מנגנוני המשחק (תזכורת — לאמת מול מקורות בזמן פיתוח)

### לוח האסטרטגיה
- 9×9 משבצות
- 5 **Power Points**: 4 פינות + מרכז
- **מחזור לומיניציטי (Luminosity Cycle)**: כל משבצת מחליפה צבע על פני זמן —
  לבן → אפור בהיר → אפור כהה → שחור → וחזרה. כלי אור חזקים על משבצות בהירות, כלי חושך על כהות.

### תנאי ניצחון
1. כיבוש כל 5 ה-Power Points בו-זמנית
2. השמדת כל הכלים של היריב
3. כליאה של היריב כך שלא יכול לזוז (lock state)

### כלים — 9 סוגים לכל צד, 18 כלים סה"כ לכל צד
**Light side**:
- Knight (חייל בסיסי, חרב)
- Archer (טווח)
- Valkyrie (מעופפת, חניתות)
- Golem (איטי, סלעים)
- Unicorn (מהיר, קרן קסם)
- Djinni (מעופף, כדורי אש)
- Phoenix (התפוצצות אש)
- Wizard (לחשים, שביר)
- King (מלך — מטיל לחשים)

**Dark side**:
- Goblin (בסיסי, חרב)
- Manticore (טווח, ארס)
- Harpy / Banshee (מעופפת)
- Troll (איטי, מתחדש)
- Basilisk (אבן)
- Dragon (מעופף, אש)
- Shapeshifter (מחקה כלי יריב)
- Sorceress (לחשים)
- Demon (מנהיג — מטיל לחשים)

(הפרמטרים המדויקים — HP, מהירות, נזק, קצב ירי — דורשים מחקר מתוך המקור)

### קרב Action
כששני כלים נפגשים, שניהם עוברים ל-**Battle Field** (זירה קטנה). שני השחקנים שולטים בכלים בזמן אמת:
- תנועה (8 כיוונים)
- ירייה / התקפה
- HP יורד בכל פגיעה
- ניצחון מי שמשמיד את היריב או דוחק אותו מהזירה
- שטח הזירה משתנה לפי צבע המשבצת המקורית

### לחשים (Wizard/Sorceress + Sovereign)
מאגר מוגבל של ~7 לחשים, כל אחד שמיש פעם אחת במהלך המשחק:
- Imprison (לכלוא כלי)
- Revive (להחיות כלי שמת)
- Teleport (להעביר כלי)
- Exchange (להחליף מקום בין 2 כלים)
- Heal (לרפא כלי)
- Summon Elemental (לזמן יסוד — אש/אדמה/מים/אוויר)
- Shift Time (להאיץ/להאט מחזור לומיניציטי)

## ארכיטקטורת המערכת

### Frontend (HTML5 Canvas + Vanilla JS)
```
/public
  /assets
    /sprites       # pixel-art sprites של כלים
    /sounds        # אפקטי קול
    /tiles         # משבצות לוח (4 דרגות בהירות)
  /js
    main.js        # Entry point + game loop
    network.js     # Socket.IO client
    board.js       # לוח אסטרטגי + רנדור
    combat.js      # מסך קרב action
    piece.js       # מחלקת כלי + סוגים
    luminosity.js  # מחזור אור/חושך
    spells.js      # לחשים
    ui.js          # תפריטים, לובי, HUD
  index.html
  style.css
```

### Backend (Node.js + Socket.IO)
```
/server
  index.js         # Express + Socket.IO entry
  /game
    GameState.js   # מצב משחק קנוני (authoritative)
    Match.js       # ניהול match בודד
    Combat.js      # רזולוציית קרב + סינכרון
    rules.js       # תנועה חוקית, ניצחון, לחשים
  /lobby
    Lobby.js       # רשימת חדרים + matchmaking
    Player.js      # מצב שחקן
package.json
```

### Networking model
- **לוח אסטרטגי**: turn-based — שרת מאשר כל מהלך
- **קרב action**: real-time — שני שחקנים שולחים inputs ~30Hz, שרת מסנכרן state ~30Hz
- **lag handling**: client-side prediction + server reconciliation

## חוקי פיתוח / Conventions
- Backend authoritative — לעולם לא לסמוך על הלקוח (אנטי-cheat בסיסי)
- אסטים: לפתח עם placeholders (מלבנים צבעוניים) קודם, ספריטים אמיתיים אחר כך
- Pixel-art: nearest-neighbor scaling, אסור anti-aliasing על הספריטים
- מבנה קוד מודולרי — כל מערכת (לוח, קרב, לחשים) קובץ נפרד
- שמירת היסטוריית מהלכים — לצרכי debug ו-replay

## תוכנית עבודה — שלבים
1. **שלד**: שרת Express מינימלי, דף HTML עם canvas, חיבור Socket.IO, הודעת hello
2. **לוח האסטרטגיה**: רנדור לוח 9×9, מחזור לומיניציטי, ספריטים placeholder ל-18 כלים, תנועה turn-based מקומית (hot-seat)
3. **קרב action — לוקאלי**: זירת קרב פשוטה, שליטה במקלדת, HP ונצחון
4. **מולטיפלייר אונליין**: lobby, matchmaking, סינכרון לוח, סינכרון קרב
5. **לחשים**: 7 לחשים, UI לבחירה, סינכרון על השרת
6. **תוכן מלא**: כל 9 סוגי הכלים לכל צד, פרמטרים מאוזנים, אנימציות
7. **ליטוש**: ספריטים HD אמיתיים, sound effects, balance, UX

## פתיחת קרב — חוויית המשתמש (UX)
1. שחקן בוחר כלי על הלוח → רואה משבצות מותרות
2. בוחר משבצת. אם ריקה → תנועה רגילה. אם תפוסה ע"י יריב → קרב
3. **טרנזישן לזירה**: fade out מהלוח, fade in לזירת קרב עם רקע תואם לצבע המשבצת
4. ספירה לאחור 3-2-1-FIGHT
5. שני הכלים נעים, יורים, חוטפים נזק עד שאחד מת או נסוג
6. המנצח חוזר ללוח, תופס את המשבצת. החדר שב למצב turn-based

## משאבים / מקורות
- [Archon — Wikipedia](https://en.wikipedia.org/wiki/Archon:_The_Light_and_the_Dark)
- [Archon Live (browser remake, נסגר ב-2020)](https://www.archonlive.com/)
- ROM/Emulator של C64 לבדיקת המקור — מומלץ להריץ על VICE emulator לעיון

## פתוחות
- האם להוסיף AI ליריב (single-player)? נדחה כרגע — multiplayer הוא ה-MVP
- האם להוסיף mobile support (touch controls)? נשקל בשלב הליטוש
- אסטים — לצייר עצמאית, להזמין, או להשתמש ב-AI image generation? פתוח
- אופציה לתפיסות חוקים שונים: original 1983 vs Archon II Adept (1984) vs custom
