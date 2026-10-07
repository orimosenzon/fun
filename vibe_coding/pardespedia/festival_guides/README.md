# מחוללי מדריכי הפסטיבלים

סקריפטים שבונים דפי מדריך בפרדספדיה מתוך התוכנית הרשמית של פסטיבל, במקום
להקליד עשרות מוקדים ביד. הועברו לכאן מתיקיות scratchpad זמניות ב-29/8/2026,
אחרי שהתברר שהן נמחקות בין סשנים.

## אמנות במושבה (pardesart.co.il)

```bash
python3 scrape_pardesart.py     # -> data/artists.json, data/foods.json
python3 scrape_events.py        # -> data/events.json   (תלוי בקודם: מייבא ממנו)
python3 build_pardesart_guide.py  # -> guide_pardesart.wiki
```

הסקריפטים קוראים את **מפות האתר** של pardesart (`artist-sitemap.xml`,
`food-sitemap.xml`, `post-sitemap.xml`), ולכן הרצה חוזרת קולטת מאליה משתתפים
שנוספו או הוסרו. אין רשימה קשיחה לתחזק.

שתי מלכודות שנפתרו וכדאי לא ליפול בהן שוב:

- **תגיות `<p>` לא נסגרות** בתבנית של האתר, ולכן `<p>(.*?)</p>` בולע את כל
  הדף. הפרסר חוסם כל פסקה ב-`<p` הבא או ב-`</p>`, מה שמגיע קודם.
- **סוג האירוע והיום** אינם בטקסט הנראה אלא ב-JSON-LD של הדף
  (`articleSection` ו-`keywords`). זה המקור האמין, לא ה-h2ים.

`data/street_centroids.json` מגיע מ-Overpass (יחס 1392849) ומשמש לחלוקה
לאזורים לפי קו אורך. ל-OSM **אין** מספרי בתים במושבה, ולכן אי אפשר לגזור
ממנו מיקום מדויק ואי אפשר להפיק מפה ממוספרת כמו זו של קהילילה לבן.

## קהילילה לבן (build_kehilila_guide.py)

בנה את [[מדריך קהילילה לבן 2026]] מתוך `data/events_geo.json`,
`data/parking.json` ו-`data/wikilinks.json`. הנתונים נאספו מאתר האירוע
ומהמפה הרשמית שלו. המפות בדף הופקו בנפרד, מתמונות מפת המושבה הרשמית שאורי
סיפק, ולא מהסקריפט הזה.

## דרך הנדיב 2026 (build_hanadiv_guide.py)

בונה את [[דרך הנדיב 2026]]. אתר הפסטיבל הוא אפליקציית Angular מעל API פתוח,
בלי מפתח: `api.hanadiv.org/festival?festival=2026` (תאריכים, קישורים) ו-
`api.hanadiv.org/event?festival=2026` (כל האירועים המאושרים, עם כתובת, שעה,
משך, קהל ונגישות). שנה אחרת = לשנות את `YEAR`.

```bash
python3 build_hanadiv_guide.py            # -> data/hanadiv_2026.json, guide_hanadiv_2026.wiki
python3 build_hanadiv_guide.py --cached   # בנייה מחדש בלי למשוך
python3 ../edit_page.py "דרך הנדיב 2026" guide_hanadiv_2026.wiki "רענון תוכנית (בוטי)" --replace
```

הגשת אירועים פתוחה עד `publicationWindowEnd` (22/10/2026), ולכן כדאי לרענן
עד אז. הכתובות מוקלדות ביד בידי המארחים: תיקונים ב-`ADDRESS_FIX`, ומוקדים
שיש להם ערך בוויקי ב-`VENUES`. את הקישורים החיצוניים של המארחים לא מעתיקים,
בכוונה: חלקם מקוצרים ומסנן הספאם יפיל את כל העריכה.
